import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Tests for the diagnosis orchestrator with red-flag integration."""

from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator, DiagnosisWithContext


def test_combine_predictions_ensemble():
    """Test that combine_predictions correctly merges Bayesian and neural predictions."""
    orchestrator = DiagnosisOrchestrator()
    
    # Sample Bayesian priors
    bayesian_probs = {
        "J45.9": 0.4,
        "J18.9": 0.3,
        "J44.9": 0.3,
    }
    
    # Sample neural predictions
    neural_predictions = [
        {"condition_code": "J45.9", "probability": 0.6},
        {"condition_code": "J18.9", "probability": 0.2},
        {"condition_code": "J44.9", "probability": 0.2},
    ]
    
    # Combine with default weights (0.6 neural, 0.4 Bayesian)
    combined = orchestrator.combine_predictions(bayesian_probs, neural_predictions)
    
    # Check that all conditions are present
    assert "J45.9" in combined
    assert "J18.9" in combined
    assert "J44.9" in combined
    
    # Check that probabilities sum to 1.0 (within tolerance)
    total = sum(combined.values())
    assert abs(total - 1.0) < 1e-6, f"Probabilities should sum to 1.0, got {total}"
    
    # Check that J45.9 has highest probability (highest in both sources)
    assert combined["J45.9"] > combined["J18.9"]
    assert combined["J45.9"] > combined["J44.9"]
    
    # Verify weighted ensemble calculation for J45.9
    # Expected: 0.6 * 0.6 + 0.4 * 0.4 = 0.36 + 0.16 = 0.52 (before normalization)
    expected_j45_unnormalized = 0.6 * 0.6 + 0.4 * 0.4
    expected_j18_unnormalized = 0.6 * 0.2 + 0.4 * 0.3
    expected_j44_unnormalized = 0.6 * 0.2 + 0.4 * 0.3
    total_unnormalized = (
        expected_j45_unnormalized + expected_j18_unnormalized + expected_j44_unnormalized
    )
    expected_j45 = expected_j45_unnormalized / total_unnormalized
    
    assert abs(combined["J45.9"] - expected_j45) < 1e-6
    
    print("✅ Ensemble calculation test passed")


def test_combine_predictions_handles_missing_conditions():
    """Test that combine_predictions handles conditions present in only one source."""
    orchestrator = DiagnosisOrchestrator()
    
    # Bayesian has J45.9, neural has J18.9
    bayesian_probs = {"J45.9": 1.0}
    neural_predictions = [{"condition_code": "J18.9", "probability": 1.0}]
    
    combined = orchestrator.combine_predictions(bayesian_probs, neural_predictions)
    
    # Both conditions should be present
    assert "J45.9" in combined
    assert "J18.9" in combined
    
    # J45.9 gets 0.4 weight (from Bayesian only)
    # J18.9 gets 0.6 weight (from neural only)
    # After normalization, J18.9 should be higher
    assert combined["J18.9"] > combined["J45.9"]
    
    # Probabilities should sum to 1.0
    total = sum(combined.values())
    assert abs(total - 1.0) < 1e-6
    
    print("✅ Missing conditions handling test passed")


def test_red_flag_detection():
    """Test that enrich_with_red_flags detects matching red-flag symptoms."""
    orchestrator = DiagnosisOrchestrator()
    
    # Patient has symptoms matching J45.9 (Asthma) red-flags
    patient_symptoms = [
        "cough",
        "severe respiratory distress",  # red-flag
        "wheezing",
        "unable to speak full sentences",  # red-flag
    ]
    
    detected = orchestrator.enrich_with_red_flags("J45.9", patient_symptoms)
    
    # Should detect at least the red-flags present
    assert len(detected) >= 2, f"Expected at least 2 red-flags, got {len(detected)}"
    assert "severe_respiratory_distress" in detected
    assert "unable_to_speak_full_sentences" in detected
    
    print("✅ Red-flag detection test passed")


def test_red_flag_detection_no_match():
    """Test that enrich_with_red_flags returns empty list when no red-flags match."""
    orchestrator = DiagnosisOrchestrator()
    
    # Patient has mild symptoms, no red-flags
    patient_symptoms = ["mild cough", "slight wheezing"]
    
    detected = orchestrator.enrich_with_red_flags("J45.9", patient_symptoms)
    
    # Should not detect any red-flags
    assert len(detected) == 0, f"Expected 0 red-flags, got {len(detected)}"
    
    print("✅ No red-flag match test passed")


def test_escalation_with_red_flags():
    """Test that red-flags trigger emergency escalation."""
    orchestrator = DiagnosisOrchestrator()
    
    # Any red-flag should trigger emergency
    red_flags = ["severe_respiratory_distress"]
    escalation = orchestrator.determine_escalation("J45.9", 0.3, red_flags)
    
    assert escalation == "emergency", f"Expected emergency, got {escalation}"
    
    print("✅ Red-flag emergency escalation test passed")


def test_escalation_high_probability_emergency_condition():
    """Test that high probability for emergency condition triggers emergency."""
    orchestrator = DiagnosisOrchestrator()
    
    # J81.0 (Acute pulmonary edema) is an emergency condition
    # High probability without red-flags should still trigger emergency
    escalation = orchestrator.determine_escalation("J81.0", 0.85, [])
    
    assert escalation == "emergency", f"Expected emergency, got {escalation}"
    
    print("✅ High probability emergency condition test passed")


def test_escalation_urgent_level():
    """Test that medium-high probability triggers urgent escalation."""
    orchestrator = DiagnosisOrchestrator()
    
    # Probability between 0.5-0.8 with no red-flags → urgent
    escalation = orchestrator.determine_escalation("J45.9", 0.6, [])
    
    assert escalation == "urgent", f"Expected urgent, got {escalation}"
    
    print("✅ Urgent escalation test passed")


def test_escalation_routine_level():
    """Test that low probability triggers routine escalation."""
    orchestrator = DiagnosisOrchestrator()
    
    # Probability below 0.5 with no red-flags → routine
    escalation = orchestrator.determine_escalation("J45.9", 0.3, [])
    
    assert escalation == "routine", f"Expected routine, got {escalation}"
    
    print("✅ Routine escalation test passed")


def test_guidance_text_generation():
    """Test that guidance text is generated for each escalation level."""
    orchestrator = DiagnosisOrchestrator()
    
    # Emergency guidance
    emergency_guidance = orchestrator.generate_guidance_text("emergency")
    assert "immediate" in emergency_guidance.lower()
    assert "911" in emergency_guidance or "emergency" in emergency_guidance.lower()
    
    # Urgent guidance
    urgent_guidance = orchestrator.generate_guidance_text("urgent")
    assert "24-48 hours" in urgent_guidance or "prompt" in urgent_guidance.lower()
    
    # Routine guidance
    routine_guidance = orchestrator.generate_guidance_text("routine")
    assert "monitor" in routine_guidance.lower()
    
    print("✅ Guidance text generation test passed")


def test_orchestrate_diagnosis_end_to_end():
    """Test the complete orchestration workflow."""
    orchestrator = DiagnosisOrchestrator()
    
    # Setup test data
    bayesian_probs = {
        "J45.9": 0.5,
        "J18.9": 0.3,
        "J44.9": 0.2,
    }
    
    neural_predictions = [
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.7},
        {"condition_code": "J18.9", "condition_name": "Pneumonia", "probability": 0.2},
        {"condition_code": "J44.9", "condition_name": "COPD", "probability": 0.1},
    ]
    
    patient_symptoms = [
        "severe respiratory distress",  # red-flag for J45.9
        "wheezing",
        "cough",
    ]
    
    # Run orchestration
    diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs, neural_predictions, patient_symptoms, top_k=3
    )
    
    # Should return 3 diagnoses
    assert len(diagnoses) == 3, f"Expected 3 diagnoses, got {len(diagnoses)}"
    
    # Check that diagnoses are DiagnosisWithContext objects
    for diagnosis in diagnoses:
        assert isinstance(diagnosis, DiagnosisWithContext)
        assert hasattr(diagnosis, "condition_code")
        assert hasattr(diagnosis, "condition_name")
        assert hasattr(diagnosis, "probability")
        assert hasattr(diagnosis, "red_flags")
        assert hasattr(diagnosis, "escalation_level")
        assert hasattr(diagnosis, "reasoning")
    
    # Top diagnosis should be J45.9 (highest in both sources)
    assert diagnoses[0].condition_code == "J45.9"
    assert diagnoses[0].condition_name == "Asthma"
    
    # J45.9 should have detected red-flags
    assert len(diagnoses[0].red_flags) > 0
    assert "severe_respiratory_distress" in diagnoses[0].red_flags
    
    # Red-flags should trigger emergency escalation
    assert diagnoses[0].escalation_level == "emergency"
    
    # Reasoning should be present
    assert len(diagnoses[0].reasoning) > 0
    assert "Asthma" in diagnoses[0].reasoning
    
    # Probabilities should be in descending order
    for i in range(len(diagnoses) - 1):
        assert diagnoses[i].probability >= diagnoses[i + 1].probability
    
    print("✅ End-to-end orchestration test passed")


def test_orchestrate_diagnosis_respects_top_k():
    """Test that orchestrate_diagnosis respects the top_k parameter."""
    orchestrator = DiagnosisOrchestrator()
    
    bayesian_probs = {
        "J45.9": 0.3,
        "J18.9": 0.25,
        "J44.9": 0.2,
        "J06.9": 0.15,
        "J20.9": 0.1,
    }
    
    neural_predictions = [
        {"condition_code": code, "condition_name": f"Condition {code}", "probability": prob}
        for code, prob in bayesian_probs.items()
    ]
    
    patient_symptoms = []
    
    # Test with top_k=2
    diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs, neural_predictions, patient_symptoms, top_k=2
    )
    
    assert len(diagnoses) == 2, f"Expected 2 diagnoses, got {len(diagnoses)}"
    
    # Test with top_k=5
    diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs, neural_predictions, patient_symptoms, top_k=5
    )
    
    assert len(diagnoses) == 5, f"Expected 5 diagnoses, got {len(diagnoses)}"
    
    print("✅ Top-k parameter test passed")


if __name__ == "__main__":
    print("🏥 Testing Diagnosis Orchestrator")
    print("=" * 50)
    
    test_combine_predictions_ensemble()
    test_combine_predictions_handles_missing_conditions()
    test_red_flag_detection()
    test_red_flag_detection_no_match()
    test_escalation_with_red_flags()
    test_escalation_high_probability_emergency_condition()
    test_escalation_urgent_level()
    test_escalation_routine_level()
    test_guidance_text_generation()
    test_orchestrate_diagnosis_end_to_end()
    test_orchestrate_diagnosis_respects_top_k()
    
    print("=" * 50)
    print("🎉 All diagnosis orchestrator tests passed!")
