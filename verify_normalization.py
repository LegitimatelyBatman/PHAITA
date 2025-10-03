#!/usr/bin/env python3
"""Verification script for symptom normalization across all modules.

This script demonstrates that the symptom normalization fix resolves the
inconsistent symptom naming issue (underscores vs spaces vs hyphens).
"""

from phaita.conversation.dialogue_engine import DialogueEngine
from phaita.models.bayesian_network import BayesianSymptomNetwork
from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator


def test_dialogue_engine_normalization():
    """Test that DialogueEngine handles all symptom formats consistently."""
    print("\n1. DialogueEngine - Symptom Format Consistency Test")
    print("   " + "=" * 56)
    
    # Test with different formats of the same symptom
    test_formats = [
        "severe_respiratory_distress",
        "severe respiratory distress",
        "Severe-Respiratory-Distress",
        "SEVERE_RESPIRATORY_DISTRESS",
    ]
    
    probability_changes = []
    
    for fmt in test_formats:
        engine = DialogueEngine()
        initial = engine.state.differential_probabilities.get("J45.9", 0)
        engine.update_beliefs(fmt, present=True)
        final = engine.state.differential_probabilities.get("J45.9", 0)
        change = final - initial
        probability_changes.append(change)
        print(f"   Format: '{fmt:35s}' → ΔP = {change:.10f}")
    
    # Check all changes are identical
    all_identical = all(
        abs(pc - probability_changes[0]) < 0.0001 for pc in probability_changes
    )
    
    if all_identical:
        print(f"   ✅ SUCCESS: All formats produce identical probability changes")
        return True
    else:
        print(f"   ❌ FAILURE: Formats produce different results")
        return False


def test_bayesian_network_normalization():
    """Test that BayesianSymptomNetwork normalizes symptom lookups."""
    print("\n2. BayesianSymptomNetwork - Symptom Lookup Consistency")
    print("   " + "=" * 56)
    
    network = BayesianSymptomNetwork()
    
    # Test different formats of the same symptom
    test_formats = [
        "shortness_of_breath",
        "shortness of breath",
        "Shortness-Of-Breath",
        "SHORTNESS OF BREATH",
    ]
    
    probabilities = []
    
    for fmt in test_formats:
        prob = network.get_symptom_probability("J45.9", fmt)
        probabilities.append(prob)
        print(f"   Format: '{fmt:30s}' → P = {prob:.2f}")
    
    # Check all probabilities are identical
    all_identical = all(abs(p - probabilities[0]) < 0.0001 for p in probabilities)
    
    if all_identical and probabilities[0] > 0:
        print(f"   ✅ SUCCESS: All formats return same probability ({probabilities[0]})")
        return True
    else:
        print(f"   ❌ FAILURE: Formats return different probabilities")
        return False


def test_red_flag_matching():
    """Test that red-flag detection works with mixed symptom formats."""
    print("\n3. DiagnosisOrchestrator - Red-Flag Detection Consistency")
    print("   " + "=" * 56)
    
    orchestrator = DiagnosisOrchestrator()
    
    # Test with different formats of red-flag symptoms
    test_cases = [
        ("underscores", ["severe_respiratory_distress", "unable_to_speak_full_sentences", "wheezing"]),
        ("spaces", ["severe respiratory distress", "unable to speak full sentences", "wheezing"]),
        ("hyphens", ["severe-respiratory-distress", "unable-to-speak-full-sentences", "wheezing"]),
        ("mixed", ["Severe_Respiratory-Distress", "Unable_To_Speak_Full-Sentences", "Wheezing"]),
    ]
    
    red_flag_counts = []
    
    for name, symptoms in test_cases:
        red_flags = orchestrator.enrich_with_red_flags("J45.9", symptoms)
        red_flag_counts.append(len(red_flags))
        print(f"   Format ({name:11s}): {len(red_flags)} red-flags detected → {red_flags}")
    
    # All should detect the same number of red-flags
    all_identical = all(count == red_flag_counts[0] for count in red_flag_counts)
    
    if all_identical and red_flag_counts[0] >= 2:
        print(f"   ✅ SUCCESS: All formats detect {red_flag_counts[0]} red-flags consistently")
        return True
    else:
        print(f"   ❌ FAILURE: Inconsistent red-flag detection")
        return False


def test_end_to_end_consistency():
    """Test complete workflow with mixed symptom formats."""
    print("\n4. End-to-End Workflow - Mixed Format Handling")
    print("   " + "=" * 56)
    
    # Scenario: Patient reports symptoms in various formats
    patient_symptoms_mixed = [
        "severe_respiratory_distress",  # underscores
        "unable to speak full sentences",  # spaces
        "Chest-Tightness",  # hyphens + caps
        "WHEEZING",  # all caps
    ]
    
    # Initialize components
    engine = DialogueEngine()
    orchestrator = DiagnosisOrchestrator()
    
    # Process symptoms through DialogueEngine
    print(f"   Processing {len(patient_symptoms_mixed)} symptoms in mixed formats...")
    for symptom in patient_symptoms_mixed:
        engine.update_beliefs(symptom, present=True)
    
    # Get diagnosis
    differential = engine.get_differential_diagnosis(top_n=3)
    top_condition = differential[0]
    
    print(f"   Top diagnosis: {top_condition['name']} (P={top_condition['probability']:.3f})")
    
    # Check red-flags
    red_flags = orchestrator.enrich_with_red_flags(
        top_condition['condition_code'],
        patient_symptoms_mixed
    )
    
    print(f"   Red-flags detected: {len(red_flags)} → {red_flags}")
    
    # Should detect red-flags despite mixed formats
    if len(red_flags) >= 1:
        print(f"   ✅ SUCCESS: System handles mixed formats correctly")
        return True
    else:
        print(f"   ❌ FAILURE: Red-flags not detected with mixed formats")
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("  PHAITA Symptom Normalization Verification")
    print("=" * 60)
    print("\nVerifying that symptoms with different formats (underscores,")
    print("spaces, hyphens, mixed case) are handled consistently across")
    print("all modules: DialogueEngine, BayesianSymptomNetwork, and")
    print("DiagnosisOrchestrator.")
    
    results = []
    
    # Run all tests
    results.append(("DialogueEngine", test_dialogue_engine_normalization()))
    results.append(("BayesianSymptomNetwork", test_bayesian_network_normalization()))
    results.append(("DiagnosisOrchestrator", test_red_flag_matching()))
    results.append(("End-to-End", test_end_to_end_consistency()))
    
    # Summary
    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = all(result for _, result in results)
    
    for module, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {module}")
    
    print("=" * 60)
    
    if all_passed:
        print("  🎉 All verification tests passed!")
        print("  ✅ Symptom normalization is working correctly across all modules.")
        print()
        return 0
    else:
        print("  ❌ Some verification tests failed.")
        print()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
