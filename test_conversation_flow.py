"""Comprehensive integration tests for multi-turn dialogue triage sessions.

Tests complete triage sessions from initial complaint through clarifying questions
to final diagnosis, including edge cases and termination conditions.
"""

from phaita.conversation.dialogue_engine import DialogueEngine
from phaita.data.icd_conditions import RespiratoryConditions


def test_complete_asthma_triage_session():
    """Test complete triage session: symptom → questions → diagnosis.
    
    Simulates a realistic asthma presentation with:
    - Initial wheezing symptom
    - 3-5 clarifying questions (may terminate early with strong evidence)
    - Final differential with top 3 diagnoses
    - Verifies correct diagnosis (asthma) appears in top 3
    """
    print("\n🏥 Testing complete asthma triage session...")
    
    # Initialize dialogue engine with slightly higher threshold to get more questions
    engine = DialogueEngine(max_turns=10, confidence_threshold=0.85)
    
    # Initial complaint: wheezing (highly indicative of asthma)
    initial_symptom = "wheezing"
    engine.update_beliefs(initial_symptom, present=True)
    engine.state.turn_count += 1
    
    # Track the dialogue
    questions_asked = []
    
    # Ask clarifying questions until termination
    for turn in range(5):  # Max 5 questions
        if engine.should_terminate():
            print(f"   ✓ Terminated after {len(questions_asked)} clarifying questions")
            break
        
        # Select next question based on information gain
        question = engine.select_next_question()
        if question is None:
            print(f"   ✓ No more questions available after {len(questions_asked)} questions")
            break
        
        questions_asked.append(question)
        
        # Simulate patient responses for asthma-related symptoms
        # Confirm: shortness_of_breath, chest_tightness
        # Deny: fever, cough_with_phlegm (more pneumonia-like)
        if question in ["shortness_of_breath", "chest_tightness"]:
            engine.answer_question(question, present=True)
        else:
            engine.answer_question(question, present=False)
    
    # Get final differential diagnosis
    differential = engine.get_differential_diagnosis(top_n=3)
    
    # Assertions
    # Note: Strong asthma evidence may cause early termination (1-5 questions)
    assert len(questions_asked) >= 1, f"Expected at least 1 question, got {len(questions_asked)}"
    assert len(questions_asked) <= 5, f"Expected at most 5 questions, got {len(questions_asked)}"
    
    assert len(differential) > 0, "Differential diagnosis should not be empty"
    assert len(differential) <= 3, f"Expected top 3, got {len(differential)}"
    
    # Check that asthma (J45.9) is in top 3
    top_codes = [d["condition_code"] for d in differential]
    assert "J45.9" in top_codes, f"Expected asthma (J45.9) in top 3, got {top_codes}"
    
    # Check that probabilities are valid and sorted
    for diag in differential:
        assert 0.0 <= diag["probability"] <= 1.0, f"Invalid probability: {diag['probability']}"
    
    probs = [d["probability"] for d in differential]
    assert probs == sorted(probs, reverse=True), "Differential should be sorted by probability"
    
    # Asthma should be the top diagnosis (or at least high probability)
    if differential[0]["condition_code"] == "J45.9":
        print(f"   ✓ Asthma correctly identified as top diagnosis")
    else:
        # If not top, verify it's at least in top 3 with decent probability
        asthma_in_top = any(d["condition_code"] == "J45.9" for d in differential)
        assert asthma_in_top, "Asthma should be in top 3"
        print(f"   ✓ Asthma correctly identified in top 3")
    
    print(f"   ✓ Asked {len(questions_asked)} questions: {questions_asked[:3]}{'...' if len(questions_asked) > 3 else ''}")
    print(f"   ✓ Top diagnosis: {differential[0]['name']} (P={differential[0]['probability']:.3f})")


def test_edge_case_deny_all_symptoms():
    """Test edge case: User denies all suggested symptoms.
    
    Verifies that the system:
    - Continues asking up to max_turns
    - Does not crash or enter invalid state
    - Eventually terminates
    - Returns a differential (uniform if all denied)
    """
    print("\n🏥 Testing edge case: deny all symptoms...")
    
    # Initialize with low max_turns for testing
    engine = DialogueEngine(max_turns=5, confidence_threshold=0.7)
    
    # Ask questions and deny everything
    questions_count = 0
    while not engine.should_terminate() and questions_count < 10:
        question = engine.select_next_question()
        if question is None:
            break
        
        # Deny all symptoms
        engine.answer_question(question, present=False)
        questions_count += 1
    
    # Should terminate due to turn limit
    assert engine.should_terminate(), "Engine should terminate after max_turns"
    assert engine.state.turn_count == 5, f"Expected 5 turns, got {engine.state.turn_count}"
    
    # All symptoms should be in denied set
    assert len(engine.state.confirmed_symptoms) == 0, "No symptoms should be confirmed"
    assert len(engine.state.denied_symptoms) >= 5, "Multiple symptoms should be denied"
    
    # Should still return a differential (though less confident)
    differential = engine.get_differential_diagnosis(top_n=3)
    assert len(differential) > 0, "Should still have differential even with all denials"
    
    # Probabilities should be relatively uniform (high entropy)
    probs = [d["probability"] for d in differential]
    max_prob = max(probs)
    assert max_prob < 0.7, f"No condition should be highly confident, got {max_prob}"
    
    print(f"   ✓ Asked {questions_count} questions, all denied")
    print(f"   ✓ Terminated after turn limit")
    print(f"   ✓ Top probability: {max_prob:.3f} (appropriately uncertain)")


def test_edge_case_conflicting_symptoms():
    """Test edge case: User confirms conflicting symptoms.
    
    Confirms symptoms that typically don't co-occur (e.g., asthma + pneumonia symptoms).
    Verifies the system handles this gracefully and still produces a differential.
    """
    print("\n🏥 Testing edge case: conflicting symptoms...")
    
    engine = DialogueEngine(max_turns=10, confidence_threshold=0.7)
    
    # Confirm conflicting symptoms:
    # Asthma indicators: wheezing, chest_tightness
    # Pneumonia indicators: fever, productive_cough, chest_pain
    conflicting_symptoms = [
        ("wheezing", True),           # Asthma
        ("chest_tightness", True),    # Asthma
        ("fever", True),              # Pneumonia
        ("productive_cough", True),   # Pneumonia
        ("chest_pain", True),         # Pneumonia
    ]
    
    for symptom, present in conflicting_symptoms:
        # Check if symptom is valid first
        all_symptoms = set()
        for cond_data in engine.conditions.values():
            all_symptoms.update(cond_data.get("symptoms", []))
            all_symptoms.update(cond_data.get("severity_indicators", []))
        
        if symptom in all_symptoms:
            engine.update_beliefs(symptom, present=present)
            engine.state.turn_count += 1
    
    # Should not crash
    differential = engine.get_differential_diagnosis(top_n=5, min_probability=0.0)
    
    # Should return multiple plausible conditions (use min_probability=0.0 to ensure results)
    assert len(differential) > 0, "Should have differential despite conflicts"
    assert len(differential) >= 1, "Should have at least one condition"
    
    # Both asthma (J45.9) and pneumonia (J18.9) might be in differential
    codes = [d["condition_code"] for d in differential]
    
    # At least one respiratory condition should be identified
    valid_conditions = ["J45.9", "J18.9", "J44.9", "J06.9", "J20.9"]
    has_valid_condition = any(code in valid_conditions for code in codes)
    assert has_valid_condition, f"Expected at least one valid condition, got {codes}"
    
    print(f"   ✓ Confirmed {len(conflicting_symptoms)} conflicting symptoms")
    print(f"   ✓ Generated differential with {len(differential)} conditions")
    print(f"   ✓ Top 3: {[d['name'] for d in differential[:3]]}")


def test_early_termination_confidence_threshold():
    """Test early termination when confidence exceeds threshold.
    
    Verifies that:
    - Dialogue terminates when top condition probability > threshold
    - Termination occurs before max_turns
    - Final diagnosis is confident (P > 0.7)
    """
    print("\n🏥 Testing early termination on confidence threshold...")
    
    # Use default confidence threshold (0.7)
    engine = DialogueEngine(max_turns=10, confidence_threshold=0.7)
    
    # Strongly suggest asthma with multiple confirming symptoms
    asthma_symptoms = ["wheezing", "shortness_of_breath", "chest_tightness"]
    
    for symptom in asthma_symptoms:
        if engine.should_terminate():
            break
        engine.update_beliefs(symptom, present=True)
        engine.state.turn_count += 1
    
    # Check termination
    assert engine.should_terminate(), "Should terminate with high confidence"
    assert engine.state.turn_count < 10, f"Should terminate early, but used {engine.state.turn_count} turns"
    
    # Get final differential
    differential = engine.get_differential_diagnosis(top_n=3)
    top_condition = differential[0]
    
    # Top condition should have high probability
    assert top_condition["probability"] > 0.7, \
        f"Top condition should be confident (>0.7), got {top_condition['probability']}"
    
    # Should be asthma given the symptoms
    assert top_condition["condition_code"] == "J45.9", \
        f"Expected asthma (J45.9), got {top_condition['condition_code']}"
    
    print(f"   ✓ Terminated early after {engine.state.turn_count} turns (max: 10)")
    print(f"   ✓ Top condition: {top_condition['name']} (P={top_condition['probability']:.3f})")
    print(f"   ✓ Confidence threshold exceeded (>0.7)")


def test_maximum_turn_limit_prevents_infinite_loops():
    """Test that maximum turn limit prevents infinite loops.
    
    Verifies:
    - Engine strictly enforces max_turns
    - Termination occurs even with ambiguous symptoms
    - System doesn't hang or loop indefinitely
    """
    print("\n🏥 Testing maximum turn limit prevents infinite loops...")
    
    # Use low turn limit for testing
    max_turns = 3
    engine = DialogueEngine(max_turns=max_turns, confidence_threshold=0.99)  # Very high threshold
    
    # Simulate ambiguous responses (alternating yes/no)
    turns_executed = 0
    question_count = 0
    
    while not engine.should_terminate() and question_count < 100:  # Safety limit
        question = engine.select_next_question()
        if question is None:
            break
        
        # Alternate between present and absent
        present = (question_count % 2 == 0)
        engine.answer_question(question, present=present)
        turns_executed += 1
        question_count += 1
    
    # Should terminate exactly at max_turns
    assert engine.should_terminate(), "Engine should terminate at max_turns"
    assert turns_executed <= max_turns, \
        f"Expected at most {max_turns} turns, executed {turns_executed}"
    assert engine.state.turn_count == max_turns, \
        f"Turn count should be {max_turns}, got {engine.state.turn_count}"
    
    # Verify no infinite loop occurred
    assert question_count < 100, "Safety limit should not be reached (indicates potential infinite loop)"
    
    print(f"   ✓ Terminated after exactly {turns_executed} turns (max: {max_turns})")
    print(f"   ✓ No infinite loop detected")
    print(f"   ✓ Turn limit strictly enforced")


def run_all_tests():
    """Run all conversation flow tests and report results."""
    tests = [
        ("Complete asthma triage session", test_complete_asthma_triage_session),
        ("Edge case: Deny all symptoms", test_edge_case_deny_all_symptoms),
        ("Edge case: Conflicting symptoms", test_edge_case_conflicting_symptoms),
        ("Early termination on confidence threshold", test_early_termination_confidence_threshold),
        ("Maximum turn limit prevents infinite loops", test_maximum_turn_limit_prevents_infinite_loops),
    ]
    
    print("🏥 PHAITA Multi-Turn Dialogue Test Suite")
    print("=" * 60)
    
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
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("🎉 All conversation flow tests passed!")
        return True
    else:
        print(f"❌ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
