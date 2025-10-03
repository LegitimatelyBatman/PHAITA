"""Comprehensive tests for escalation guidance and routing.

Tests emergency red-flag detection, urgent care routing, routine guidance,
and condition-specific action recommendations.
"""

from phaita.conversation.dialogue_engine import DialogueEngine
from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator, DiagnosisWithContext


def test_emergency_red_flag_triggers_guidance():
    """Test that emergency red-flags trigger correct emergency guidance.
    
    Verifies:
    - Red-flag symptoms are detected
    - Escalation level is set to 'emergency'
    - Guidance includes emergency keywords (911, immediate, emergency room)
    """
    print("\n🚑 Testing emergency red-flag detection and guidance...")
    
    orchestrator = DiagnosisOrchestrator()
    
    # Patient with asthma and red-flag symptoms
    patient_symptoms = [
        "wheezing",
        "severe_respiratory_distress",      # RED FLAG
        "unable_to_speak_full_sentences",   # RED FLAG
        "chest_tightness",
    ]
    
    # Bayesian probabilities (asthma is most likely)
    bayesian_probs = {
        "J45.9": 0.6,  # Asthma
        "J44.9": 0.2,  # COPD
        "J18.9": 0.2,  # Pneumonia
    }
    
    # Neural predictions (asthma is most likely)
    neural_predictions = [
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.7},
        {"condition_code": "J44.9", "condition_name": "COPD", "probability": 0.2},
        {"condition_code": "J18.9", "condition_name": "Pneumonia", "probability": 0.1},
    ]
    
    # Run orchestration
    diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs, neural_predictions, patient_symptoms, top_k=3
    )
    
    # Find asthma diagnosis
    asthma_diagnosis = None
    for diag in diagnoses:
        if diag.condition_code == "J45.9":
            asthma_diagnosis = diag
            break
    
    assert asthma_diagnosis is not None, "Asthma should be in differential"
    
    # Check red-flag detection
    assert len(asthma_diagnosis.red_flags) >= 2, \
        f"Expected at least 2 red-flags, got {len(asthma_diagnosis.red_flags)}"
    assert "severe_respiratory_distress" in asthma_diagnosis.red_flags, \
        "Should detect severe respiratory distress"
    assert "unable_to_speak_full_sentences" in asthma_diagnosis.red_flags, \
        "Should detect unable to speak full sentences"
    
    # Check escalation level
    assert asthma_diagnosis.escalation_level == "emergency", \
        f"Expected emergency escalation, got {asthma_diagnosis.escalation_level}"
    
    # Generate and check guidance text
    guidance = orchestrator.generate_guidance_text("emergency")
    guidance_lower = guidance.lower()
    
    assert any(keyword in guidance_lower for keyword in ["911", "immediate", "emergency"]), \
        f"Guidance should include emergency keywords: {guidance}"
    
    print(f"   ✓ Detected {len(asthma_diagnosis.red_flags)} red-flags: {asthma_diagnosis.red_flags}")
    print(f"   ✓ Escalation level: {asthma_diagnosis.escalation_level}")
    print(f"   ✓ Guidance includes emergency keywords")


def test_urgent_care_routing_moderate_cases():
    """Test urgent care routing for moderate probability cases.
    
    Verifies:
    - Moderate probability (0.5-0.8) without red-flags → urgent
    - Guidance recommends prompt care (24-48 hours)
    - No emergency keywords in guidance
    """
    print("\n⚠️  Testing urgent care routing for moderate cases...")
    
    orchestrator = DiagnosisOrchestrator()
    
    # Patient with moderate asthma symptoms (no red-flags)
    patient_symptoms = [
        "wheezing",
        "mild_shortness_of_breath",
        "occasional_chest_tightness",
    ]
    
    # Moderate probability for asthma
    bayesian_probs = {
        "J45.9": 0.5,  # Asthma (moderate)
        "J20.9": 0.3,  # Bronchitis
        "J06.9": 0.2,  # URI
    }
    
    neural_predictions = [
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.6},
        {"condition_code": "J20.9", "condition_name": "Acute Bronchitis", "probability": 0.3},
        {"condition_code": "J06.9", "condition_name": "Upper Respiratory Infection", "probability": 0.1},
    ]
    
    # Run orchestration
    diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs, neural_predictions, patient_symptoms, top_k=3
    )
    
    # Get top diagnosis
    top_diagnosis = diagnoses[0]
    
    # Should be asthma with no red-flags
    assert top_diagnosis.condition_code == "J45.9", \
        f"Expected asthma (J45.9), got {top_diagnosis.condition_code}"
    assert len(top_diagnosis.red_flags) == 0, \
        f"Expected no red-flags, got {top_diagnosis.red_flags}"
    
    # Should be urgent escalation (probability >= 0.5, no red-flags)
    assert top_diagnosis.escalation_level == "urgent", \
        f"Expected urgent escalation, got {top_diagnosis.escalation_level}"
    
    # Check guidance text
    guidance = orchestrator.generate_guidance_text("urgent")
    guidance_lower = guidance.lower()
    
    assert "24-48 hours" in guidance or "prompt" in guidance_lower, \
        f"Urgent guidance should mention timeframe: {guidance}"
    assert "911" not in guidance and "emergency" not in guidance_lower, \
        f"Urgent guidance should not include emergency keywords: {guidance}"
    
    print(f"   ✓ Top diagnosis: {top_diagnosis.condition_name} (P={top_diagnosis.probability:.3f})")
    print(f"   ✓ No red-flags detected")
    print(f"   ✓ Escalation level: {top_diagnosis.escalation_level}")
    print(f"   ✓ Guidance recommends prompt care (24-48 hours)")


def test_routine_care_guidance_low_probability():
    """Test routine care guidance for low probability cases.
    
    Verifies:
    - Low probability (<0.5) without red-flags → routine
    - Guidance includes monitoring and self-care advice
    - No urgent or emergency keywords
    """
    print("\n🏥 Testing routine care guidance for low probability cases...")
    
    orchestrator = DiagnosisOrchestrator()
    
    # Patient with mild symptoms
    patient_symptoms = [
        "mild_cough",
        "slight_fatigue",
    ]
    
    # Low probabilities across the board
    bayesian_probs = {
        "J06.9": 0.3,  # URI
        "J20.9": 0.25, # Bronchitis
        "J45.9": 0.2,  # Asthma
        "J44.9": 0.15, # COPD
        "J18.9": 0.1,  # Pneumonia
    }
    
    neural_predictions = [
        {"condition_code": "J06.9", "condition_name": "Upper Respiratory Infection", "probability": 0.35},
        {"condition_code": "J20.9", "condition_name": "Acute Bronchitis", "probability": 0.25},
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.2},
    ]
    
    # Run orchestration
    diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs, neural_predictions, patient_symptoms, top_k=3
    )
    
    # Get top diagnosis
    top_diagnosis = diagnoses[0]
    
    # Should have low probability
    assert top_diagnosis.probability < 0.5, \
        f"Expected low probability (<0.5), got {top_diagnosis.probability}"
    
    # Should have no red-flags
    assert len(top_diagnosis.red_flags) == 0, \
        f"Expected no red-flags, got {top_diagnosis.red_flags}"
    
    # Should be routine escalation
    assert top_diagnosis.escalation_level == "routine", \
        f"Expected routine escalation, got {top_diagnosis.escalation_level}"
    
    # Check guidance text
    guidance = orchestrator.generate_guidance_text("routine")
    guidance_lower = guidance.lower()
    
    assert "monitor" in guidance_lower, \
        f"Routine guidance should mention monitoring: {guidance}"
    assert any(keyword in guidance_lower for keyword in ["rest", "hydration"]), \
        f"Routine guidance should include self-care advice: {guidance}"
    assert "911" not in guidance and "emergency" not in guidance_lower, \
        f"Routine guidance should not include emergency keywords: {guidance}"
    
    print(f"   ✓ Top diagnosis: {top_diagnosis.condition_name} (P={top_diagnosis.probability:.3f})")
    print(f"   ✓ No red-flags detected")
    print(f"   ✓ Escalation level: {top_diagnosis.escalation_level}")
    print(f"   ✓ Guidance includes monitoring and self-care")


def test_guidance_text_includes_condition_specific_actions():
    """Test that guidance text includes condition-specific actions.
    
    Verifies:
    - Each escalation level has distinct guidance
    - Emergency guidance mentions specific actions (call 911, ER)
    - Urgent guidance mentions specific timeframe (24-48 hours)
    - Routine guidance mentions specific actions (monitor, hydrate, rest)
    """
    print("\n📋 Testing condition-specific action guidance...")
    
    orchestrator = DiagnosisOrchestrator()
    
    # Test all three escalation levels
    escalation_levels = ["emergency", "urgent", "routine"]
    
    for level in escalation_levels:
        guidance = orchestrator.generate_guidance_text(level)
        
        # Verify guidance is not empty
        assert len(guidance) > 0, f"Guidance for {level} should not be empty"
        
        # Verify guidance is descriptive (at least 50 characters)
        assert len(guidance) >= 50, \
            f"Guidance for {level} should be descriptive (>50 chars), got {len(guidance)}"
        
        # Check for level-specific keywords
        guidance_lower = guidance.lower()
        
        if level == "emergency":
            # Should mention immediate action and emergency services
            assert any(keyword in guidance_lower for keyword in ["911", "emergency room", "immediate"]), \
                f"Emergency guidance should mention emergency services: {guidance}"
            assert any(keyword in guidance_lower for keyword in ["severe", "life-threatening", "chest pain", "breathing"]), \
                f"Emergency guidance should mention serious symptoms: {guidance}"
        
        elif level == "urgent":
            # Should mention prompt care timeframe
            assert any(keyword in guidance_lower for keyword in ["24-48 hours", "prompt", "soon"]), \
                f"Urgent guidance should mention timeframe: {guidance}"
            assert "doctor" in guidance_lower or "appointment" in guidance_lower, \
                f"Urgent guidance should mention doctor/appointment: {guidance}"
        
        elif level == "routine":
            # Should mention monitoring and self-care
            assert "monitor" in guidance_lower, \
                f"Routine guidance should mention monitoring: {guidance}"
            assert any(keyword in guidance_lower for keyword in ["rest", "hydration", "hydrate"]), \
                f"Routine guidance should mention self-care: {guidance}"
            assert any(keyword in guidance_lower for keyword in ["worsen", "persist"]), \
                f"Routine guidance should mention when to escalate: {guidance}"
        
        print(f"   ✓ {level.capitalize()} guidance: {len(guidance)} chars, includes required keywords")
    
    print(f"   ✓ All escalation levels have distinct, condition-specific guidance")


def test_pneumonia_red_flags_trigger_emergency():
    """Test that pneumonia-specific red-flags trigger emergency escalation.
    
    Verifies:
    - Pneumonia red-flags (confusion, low O2) are detected
    - Escalation is set to emergency
    - Different red-flags than asthma
    """
    print("\n🚑 Testing pneumonia-specific red-flag detection...")
    
    orchestrator = DiagnosisOrchestrator()
    
    # Patient with pneumonia and specific red-flags
    patient_symptoms = [
        "productive_cough",
        "fever",
        "confusion",                    # RED FLAG for pneumonia
        "oxygen_saturation_below_92",   # RED FLAG for pneumonia
        "chest_pain",
    ]
    
    # Pneumonia is most likely
    bayesian_probs = {
        "J18.9": 0.7,  # Pneumonia
        "J15.9": 0.2,  # Bacterial pneumonia
        "J45.9": 0.1,  # Asthma
    }
    
    neural_predictions = [
        {"condition_code": "J18.9", "condition_name": "Pneumonia", "probability": 0.8},
        {"condition_code": "J15.9", "condition_name": "Bacterial Pneumonia", "probability": 0.15},
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.05},
    ]
    
    # Run orchestration
    diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs, neural_predictions, patient_symptoms, top_k=3
    )
    
    # Get pneumonia diagnosis
    pneumonia_diagnosis = diagnoses[0]
    
    assert pneumonia_diagnosis.condition_code == "J18.9", \
        f"Expected pneumonia (J18.9), got {pneumonia_diagnosis.condition_code}"
    
    # Check red-flag detection
    assert len(pneumonia_diagnosis.red_flags) >= 1, \
        f"Expected red-flags, got {len(pneumonia_diagnosis.red_flags)}"
    
    # Should detect pneumonia-specific red-flags
    detected_flags_normalized = [flag.lower().replace("_", " ") for flag in pneumonia_diagnosis.red_flags]
    assert any("confusion" in flag for flag in detected_flags_normalized), \
        f"Should detect confusion, got {pneumonia_diagnosis.red_flags}"
    
    # Check escalation
    assert pneumonia_diagnosis.escalation_level == "emergency", \
        f"Expected emergency escalation, got {pneumonia_diagnosis.escalation_level}"
    
    print(f"   ✓ Detected pneumonia-specific red-flags: {pneumonia_diagnosis.red_flags}")
    print(f"   ✓ Escalation level: {pneumonia_diagnosis.escalation_level}")


def test_high_probability_emergency_condition_without_red_flags():
    """Test that high probability for emergency conditions triggers emergency even without red-flags.
    
    Verifies:
    - Emergency conditions (J81.0, J93.0) with high probability → emergency
    - Even without explicit red-flag symptoms present
    """
    print("\n🚨 Testing high probability emergency condition...")
    
    orchestrator = DiagnosisOrchestrator()
    
    # Test with acute pulmonary edema (J81.0) - emergency condition
    patient_symptoms = [
        "severe_breathlessness",
        "cough",
    ]
    
    # High probability for emergency condition
    bayesian_probs = {
        "J81.0": 0.85,  # Acute pulmonary edema (emergency)
        "J45.9": 0.1,
        "J18.9": 0.05,
    }
    
    neural_predictions = [
        {"condition_code": "J81.0", "condition_name": "Acute Pulmonary Edema", "probability": 0.9},
        {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.08},
        {"condition_code": "J18.9", "condition_name": "Pneumonia", "probability": 0.02},
    ]
    
    # Run orchestration (no red-flag symptoms explicitly listed)
    diagnoses = orchestrator.orchestrate_diagnosis(
        bayesian_probs, neural_predictions, patient_symptoms, top_k=3
    )
    
    # Get top diagnosis
    top_diagnosis = diagnoses[0]
    
    assert top_diagnosis.condition_code == "J81.0", \
        f"Expected J81.0, got {top_diagnosis.condition_code}"
    
    # Should be emergency due to high probability + emergency condition
    assert top_diagnosis.escalation_level == "emergency", \
        f"Expected emergency for high-probability emergency condition, got {top_diagnosis.escalation_level}"
    
    print(f"   ✓ Condition: {top_diagnosis.condition_name} (P={top_diagnosis.probability:.3f})")
    print(f"   ✓ Escalation level: {top_diagnosis.escalation_level}")
    print(f"   ✓ Emergency triggered by condition type + high probability")


def run_all_tests():
    """Run all escalation guidance tests and report results."""
    tests = [
        ("Emergency red-flag triggers guidance", test_emergency_red_flag_triggers_guidance),
        ("Urgent care routing for moderate cases", test_urgent_care_routing_moderate_cases),
        ("Routine care guidance for low probability", test_routine_care_guidance_low_probability),
        ("Guidance text includes condition-specific actions", test_guidance_text_includes_condition_specific_actions),
        ("Pneumonia red-flags trigger emergency", test_pneumonia_red_flags_trigger_emergency),
        ("High probability emergency condition", test_high_probability_emergency_condition_without_red_flags),
    ]
    
    print("🏥 PHAITA Escalation Guidance Test Suite")
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
        print("🎉 All escalation guidance tests passed!")
        return True
    else:
        print(f"❌ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
