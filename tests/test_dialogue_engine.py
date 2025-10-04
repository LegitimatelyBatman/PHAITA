import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Tests for the dialogue engine with Bayesian belief updating."""

import math

from phaita.conversation.dialogue_engine import DialogueEngine, DialogueState
from phaita.data.icd_conditions import RespiratoryConditions


def test_dialogue_state_initialization():
    """Test that DialogueState initializes correctly."""
    state = DialogueState()
    
    assert state.differential_probabilities == {}
    assert state.asked_questions == []
    assert state.confirmed_symptoms == set()
    assert state.denied_symptoms == set()
    assert state.turn_count == 0
    assert state.confidence_threshold == 0.7


def test_dialogue_engine_initialization():
    """Test that DialogueEngine initializes with uniform priors."""
    engine = DialogueEngine()
    
    # Should have uniform probabilities
    probs = list(engine.state.differential_probabilities.values())
    assert len(probs) > 0
    assert all(abs(p - probs[0]) < 0.001 for p in probs)  # All approximately equal
    assert abs(sum(probs) - 1.0) < 0.001  # Sum to 1.0


def test_belief_updating_with_positive_evidence():
    """Test that belief updating works correctly with positive evidence."""
    engine = DialogueEngine()
    
    # Get initial probability for asthma (J45.9)
    initial_prob = engine.state.differential_probabilities.get("J45.9", 0)
    
    # Update with wheezing (a primary symptom of asthma)
    engine.update_beliefs("wheezing", present=True)
    
    # Check that wheezing is marked as confirmed
    assert "wheezing" in engine.state.confirmed_symptoms
    assert "wheezing" not in engine.state.denied_symptoms
    
    # Asthma probability should increase (wheezing is highly associated with asthma)
    updated_prob = engine.state.differential_probabilities.get("J45.9", 0)
    assert updated_prob > initial_prob
    
    # Probabilities should still sum to 1.0
    assert abs(sum(engine.state.differential_probabilities.values()) - 1.0) < 0.001


def test_belief_updating_with_negative_evidence():
    """Test that belief updating works correctly with negative evidence."""
    engine = DialogueEngine()
    
    # Get initial probability for asthma (J45.9)
    initial_prob = engine.state.differential_probabilities.get("J45.9", 0)
    
    # Update with wheezing absent (wheezing is a primary symptom of asthma)
    engine.update_beliefs("wheezing", present=False)
    
    # Check that wheezing is marked as denied
    assert "wheezing" not in engine.state.confirmed_symptoms
    assert "wheezing" in engine.state.denied_symptoms
    
    # Asthma probability should decrease
    updated_prob = engine.state.differential_probabilities.get("J45.9", 0)
    assert updated_prob < initial_prob
    
    # Probabilities should still sum to 1.0
    assert abs(sum(engine.state.differential_probabilities.values()) - 1.0) < 0.001


def test_belief_updating_multiple_symptoms():
    """Test belief updating with multiple pieces of evidence."""
    engine = DialogueEngine()
    
    initial_prob = engine.state.differential_probabilities.get("J45.9", 0)
    
    # Update with multiple symptoms characteristic of asthma
    engine.update_beliefs("wheezing", present=True)
    engine.update_beliefs("shortness_of_breath", present=True)
    engine.update_beliefs("chest_tightness", present=True)
    
    # Asthma probability should be significantly higher
    final_prob = engine.state.differential_probabilities.get("J45.9", 0)
    assert final_prob > initial_prob
    
    # Check all symptoms are tracked
    assert len(engine.state.confirmed_symptoms) == 3
    assert "wheezing" in engine.state.confirmed_symptoms
    assert "shortness_of_breath" in engine.state.confirmed_symptoms
    assert "chest_tightness" in engine.state.confirmed_symptoms


def test_termination_high_confidence():
    """Test termination when top condition exceeds confidence threshold."""
    # Create engine with custom initial prior (simulate high confidence)
    conditions = RespiratoryConditions.get_all_conditions()
    initial_prior = {code: 0.01 for code in conditions.keys()}
    initial_prior["J45.9"] = 0.91  # Very high probability for asthma
    
    engine = DialogueEngine(initial_prior=initial_prior, confidence_threshold=0.7)
    
    # Should terminate immediately due to high confidence
    assert engine.should_terminate()


def test_termination_top_three_sum():
    """Test termination when top 3 conditions sum to > 0.9."""
    conditions = RespiratoryConditions.get_all_conditions()
    condition_codes = list(conditions.keys())
    
    # Create prior where top 3 sum to 0.95
    initial_prior = {code: 0.01 for code in condition_codes}
    if len(condition_codes) >= 3:
        initial_prior[condition_codes[0]] = 0.4
        initial_prior[condition_codes[1]] = 0.3
        initial_prior[condition_codes[2]] = 0.25
    
    # Normalize
    total = sum(initial_prior.values())
    initial_prior = {k: v / total for k, v in initial_prior.items()}
    
    engine = DialogueEngine(initial_prior=initial_prior)
    
    # Should terminate because top 3 sum > 0.9
    assert engine.should_terminate()


def test_termination_turn_limit():
    """Test termination when turn limit is reached."""
    engine = DialogueEngine(max_turns=3)
    
    # Should not terminate initially
    assert not engine.should_terminate()
    
    # Simulate 3 turns
    engine.state.turn_count = 3
    
    # Should terminate at turn limit
    assert engine.should_terminate()


def test_question_repetition_prevention():
    """Test that questions are not repeated."""
    engine = DialogueEngine()
    
    # Select first question
    first_question = engine.select_next_question()
    assert first_question is not None
    
    # Question should be in asked_questions
    assert first_question in engine.state.asked_questions
    
    # Select second question
    second_question = engine.select_next_question()
    
    # Should be different from first
    assert second_question != first_question
    
    # Both should be in asked_questions
    assert first_question in engine.state.asked_questions
    assert second_question in engine.state.asked_questions


def test_information_gain_calculation():
    """Test that information gain is calculated for symptoms."""
    engine = DialogueEngine()
    
    # Calculate information gain for a symptom
    # This should not crash and should return a reasonable value
    gain = engine._calculate_information_gain("wheezing")
    
    assert isinstance(gain, float)
    assert gain >= 0.0  # Information gain should be non-negative
    
    # After updating beliefs, information gain for the same symptom should change
    engine.update_beliefs("cough", present=True)
    new_gain = engine._calculate_information_gain("wheezing")
    
    # Gain might be different after updating beliefs
    assert isinstance(new_gain, float)
    assert new_gain >= 0.0


def test_information_gain_decreases_with_confidence():
    """Test that information gain generally decreases as confidence increases."""
    conditions = RespiratoryConditions.get_all_conditions()
    
    # Start with uniform prior
    engine1 = DialogueEngine()
    
    # Start with high-confidence prior
    initial_prior = {code: 0.01 for code in conditions.keys()}
    initial_prior["J45.9"] = 0.91
    engine2 = DialogueEngine(initial_prior=initial_prior)
    
    # Calculate entropy for both
    probs1 = list(engine1.state.differential_probabilities.values())
    probs2 = list(engine2.state.differential_probabilities.values())
    
    entropy1 = engine1._calculate_entropy(probs1)
    entropy2 = engine2._calculate_entropy(probs2)
    
    # High-confidence case should have lower entropy
    assert entropy2 < entropy1


def test_select_next_question_exhaustion():
    """Test that select_next_question returns None when all symptoms asked."""
    engine = DialogueEngine()
    
    # Collect all symptoms
    all_symptoms = set()
    for condition_data in engine.conditions.values():
        all_symptoms.update(condition_data.get("symptoms", []))
        all_symptoms.update(condition_data.get("severity_indicators", []))
    
    # Mark all as asked
    engine.state.asked_questions = list(all_symptoms)
    
    # Should return None
    next_question = engine.select_next_question()
    assert next_question is None


def test_get_differential_diagnosis():
    """Test getting the differential diagnosis."""
    engine = DialogueEngine()
    
    # Update with some evidence
    engine.update_beliefs("wheezing", present=True)
    
    # Get differential
    differential = engine.get_differential_diagnosis()
    
    assert isinstance(differential, list)
    assert len(differential) > 0
    
    # Check structure
    for entry in differential:
        assert "condition_code" in entry
        assert "name" in entry
        assert "probability" in entry
        assert isinstance(entry["probability"], float)
        assert 0.0 <= entry["probability"] <= 1.0
    
    # Should be sorted by probability (descending)
    probs = [entry["probability"] for entry in differential]
    assert probs == sorted(probs, reverse=True)


def test_get_differential_diagnosis_filtering():
    """Test that low-probability conditions are filtered out."""
    conditions = RespiratoryConditions.get_all_conditions()
    condition_codes = list(conditions.keys())
    
    # Create prior with one high probability and others very low
    initial_prior = {code: 0.001 for code in condition_codes}
    initial_prior[condition_codes[0]] = 0.99
    
    # Normalize
    total = sum(initial_prior.values())
    initial_prior = {k: v / total for k, v in initial_prior.items()}
    
    engine = DialogueEngine(initial_prior=initial_prior)
    
    # Get differential with min_probability filter
    differential = engine.get_differential_diagnosis(min_probability=0.01)
    
    # Should only include conditions above threshold
    assert all(entry["probability"] >= 0.01 for entry in differential)


def test_get_differential_diagnosis_top_n():
    """Test that top_n parameter limits results."""
    engine = DialogueEngine()
    
    # Get only top 3
    differential = engine.get_differential_diagnosis(top_n=3)
    
    assert len(differential) <= 3


def test_answer_question_increments_turn_count():
    """Test that answer_question increments turn count."""
    engine = DialogueEngine()
    
    initial_count = engine.state.turn_count
    
    engine.answer_question("wheezing", present=True)
    
    assert engine.state.turn_count == initial_count + 1


def test_reset():
    """Test that reset restores initial state."""
    engine = DialogueEngine()
    
    # Modify state
    engine.update_beliefs("wheezing", present=True)
    engine.state.turn_count = 5
    engine.state.asked_questions.append("cough")
    
    # Reset
    engine.reset()
    
    # Should be back to initial state
    assert engine.state.turn_count == 0
    assert len(engine.state.asked_questions) == 0
    assert len(engine.state.confirmed_symptoms) == 0
    assert len(engine.state.denied_symptoms) == 0
    
    # Probabilities should be uniform again
    probs = list(engine.state.differential_probabilities.values())
    assert all(abs(p - probs[0]) < 0.001 for p in probs)


def test_symptom_evidence_switching():
    """Test that symptom evidence can be updated (switch from confirmed to denied)."""
    engine = DialogueEngine()
    
    # First say symptom is present
    engine.update_beliefs("wheezing", present=True)
    assert "wheezing" in engine.state.confirmed_symptoms
    assert "wheezing" not in engine.state.denied_symptoms
    
    # Then say it's absent (correction)
    engine.update_beliefs("wheezing", present=False)
    assert "wheezing" not in engine.state.confirmed_symptoms
    assert "wheezing" in engine.state.denied_symptoms


def test_numerical_stability_repeated_denials():
    """Test numerical stability when patient denies all symptoms repeatedly."""
    engine = DialogueEngine()
    
    # Collect a lot of symptoms
    all_symptoms = set()
    for condition_data in engine.conditions.values():
        all_symptoms.update(condition_data.get("symptoms", []))
        all_symptoms.update(condition_data.get("severity_indicators", []))
    
    # Deny many symptoms (this could cause underflow without safeguards)
    symptoms_to_deny = list(all_symptoms)[:20]  # Deny 20 symptoms
    
    for symptom in symptoms_to_deny:
        engine.update_beliefs(symptom, present=False)
    
    # Probabilities should still be valid (no NaN, no 0 for all)
    probs = list(engine.state.differential_probabilities.values())
    
    # Check no NaN values
    assert all(not math.isnan(p) for p in probs), "Found NaN probability"
    
    # Check probabilities sum to approximately 1.0
    total = sum(probs)
    assert abs(total - 1.0) < 0.001, f"Probabilities should sum to 1.0, got {total}"
    
    # Check all probabilities are positive
    assert all(p > 0 for p in probs), "All probabilities should be positive"
    
    # Check probabilities are reasonable (not too extreme)
    assert all(p < 1.0 for p in probs), "No single probability should be 1.0"


def test_numerical_stability_underflow_recovery():
    """Test that underflow triggers uniform distribution reset."""
    engine = DialogueEngine()
    
    # Manually create a near-underflow scenario
    # Set all probabilities to extremely small values
    for code in engine.state.differential_probabilities:
        engine.state.differential_probabilities[code] = 1e-15
    
    # Try to normalize - should trigger underflow detection and reset
    engine._normalize_probabilities()
    
    # Should have been reset to uniform distribution
    probs = list(engine.state.differential_probabilities.values())
    n = len(probs)
    expected_uniform = 1.0 / n
    
    # All probabilities should be approximately equal
    assert all(abs(p - expected_uniform) < 0.001 for p in probs), \
        "Underflow should trigger reset to uniform distribution"
    
    # Should sum to 1.0
    assert abs(sum(probs) - 1.0) < 0.001


def test_likelihood_bounds_enforced():
    """Test that likelihood bounds prevent extreme values."""
    engine = DialogueEngine()
    
    # Update with a symptom - the bounds should be applied
    initial_probs = dict(engine.state.differential_probabilities)
    engine.update_beliefs("wheezing", present=True)
    
    # Probabilities should have changed but remain valid
    probs = list(engine.state.differential_probabilities.values())
    
    # No probability should be 0 or 1 due to bounds
    assert all(0 < p < 1 for p in probs), \
        "Likelihood bounds should prevent probabilities of exactly 0 or 1"
    
    # Test with absence
    engine.reset()
    engine.update_beliefs("wheezing", present=False)
    
    probs = list(engine.state.differential_probabilities.values())
    assert all(0 < p < 1 for p in probs), \
        "Likelihood bounds should prevent probabilities of exactly 0 or 1 (absence case)"


def test_conflicting_evidence_stability():
    """Test stability with conflicting symptom evidence."""
    engine = DialogueEngine()
    
    # Confirm symptom
    engine.update_beliefs("wheezing", present=True)
    prob_after_confirm = dict(engine.state.differential_probabilities)
    
    # Deny the same symptom (conflicting)
    engine.update_beliefs("wheezing", present=False)
    prob_after_deny = dict(engine.state.differential_probabilities)
    
    # Probabilities should still be valid
    probs = list(prob_after_deny.values())
    assert all(not math.isnan(p) for p in probs), "No NaN values"
    assert abs(sum(probs) - 1.0) < 0.001, "Should sum to 1.0"
    assert all(p > 0 for p in probs), "All probabilities positive"


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("DialogueState initialization", test_dialogue_state_initialization),
        ("DialogueEngine initialization", test_dialogue_engine_initialization),
        ("Belief updating with positive evidence", test_belief_updating_with_positive_evidence),
        ("Belief updating with negative evidence", test_belief_updating_with_negative_evidence),
        ("Belief updating with multiple symptoms", test_belief_updating_multiple_symptoms),
        ("Termination on high confidence", test_termination_high_confidence),
        ("Termination on top 3 sum", test_termination_top_three_sum),
        ("Termination on turn limit", test_termination_turn_limit),
        ("Question repetition prevention", test_question_repetition_prevention),
        ("Information gain calculation", test_information_gain_calculation),
        ("Information gain decreases with confidence", test_information_gain_decreases_with_confidence),
        ("Select next question exhaustion", test_select_next_question_exhaustion),
        ("Get differential diagnosis", test_get_differential_diagnosis),
        ("Get differential diagnosis filtering", test_get_differential_diagnosis_filtering),
        ("Get differential diagnosis top_n", test_get_differential_diagnosis_top_n),
        ("Answer question increments turn count", test_answer_question_increments_turn_count),
        ("Reset functionality", test_reset),
        ("Symptom evidence switching", test_symptom_evidence_switching),
        ("Numerical stability with repeated denials", test_numerical_stability_repeated_denials),
        ("Numerical stability underflow recovery", test_numerical_stability_underflow_recovery),
        ("Likelihood bounds enforced", test_likelihood_bounds_enforced),
        ("Conflicting evidence stability", test_conflicting_evidence_stability),
    ]
    
    print("🏥 PHAITA Dialogue Engine Test Suite")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {test_name}: {type(e).__name__}: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"❌ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
