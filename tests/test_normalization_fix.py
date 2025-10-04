import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

#!/usr/bin/env python3
"""Test that demonstrates the fix for inconsistent symptom naming.

This test specifically verifies the problem described in the issue:
"Inconsistent symptom naming (underscores vs spaces vs hyphens) causes 
red-flag matching failures"

It tests the exact scenario from the problem statement where symptoms
like 'severe_respiratory_distress' and 'severe respiratory distress'
should match identically.
"""

from phaita.conversation.dialogue_engine import DialogueEngine
from phaita.models.bayesian_network import BayesianSymptomNetwork
from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator


def test_problem_statement_scenario():
    """Test the exact scenario from the problem statement."""
    print("\n" + "=" * 70)
    print("  Testing Problem Statement Scenario")
    print("=" * 70)
    print("\nProblem: 'severe_respiratory_distress' and 'severe respiratory distress'")
    print("should match identically but previously didn't due to inconsistent")
    print("symptom name normalization.\n")
    
    # Test Case 1: DialogueEngine should treat both formats identically
    print("Test 1: DialogueEngine belief updating")
    print("-" * 70)
    
    engine1 = DialogueEngine()
    initial1 = engine1.state.differential_probabilities.get("J45.9", 0)
    engine1.update_beliefs("severe_respiratory_distress", present=True)
    final1 = engine1.state.differential_probabilities.get("J45.9", 0)
    change1 = final1 - initial1
    
    engine2 = DialogueEngine()
    initial2 = engine2.state.differential_probabilities.get("J45.9", 0)
    engine2.update_beliefs("severe respiratory distress", present=True)
    final2 = engine2.state.differential_probabilities.get("J45.9", 0)
    change2 = final2 - initial2
    
    print(f"  'severe_respiratory_distress' → Probability change: {change1:.10f}")
    print(f"  'severe respiratory distress' → Probability change: {change2:.10f}")
    print(f"  Difference: {abs(change1 - change2):.15f}")
    
    test1_passed = abs(change1 - change2) < 0.0001
    print(f"  {'✅ PASS' if test1_passed else '❌ FAIL'}: Both formats produce identical results\n")
    
    # Test Case 2: BayesianSymptomNetwork should return same probability
    print("Test 2: BayesianSymptomNetwork probability lookup")
    print("-" * 70)
    
    network = BayesianSymptomNetwork()
    prob1 = network.get_symptom_probability("J45.9", "severe_respiratory_distress")
    prob2 = network.get_symptom_probability("J45.9", "severe respiratory distress")
    
    print(f"  'severe_respiratory_distress' → Probability: {prob1}")
    print(f"  'severe respiratory distress' → Probability: {prob2}")
    print(f"  Difference: {abs(prob1 - prob2)}")
    
    test2_passed = abs(prob1 - prob2) < 0.0001
    print(f"  {'✅ PASS' if test2_passed else '❌ FAIL'}: Both formats return same probability\n")
    
    # Test Case 3: DiagnosisOrchestrator red-flag detection should work with both
    print("Test 3: DiagnosisOrchestrator red-flag detection")
    print("-" * 70)
    
    orchestrator = DiagnosisOrchestrator()
    
    # Test with underscore format
    symptoms1 = ["severe_respiratory_distress", "wheezing"]
    red_flags1 = orchestrator.enrich_with_red_flags("J45.9", symptoms1)
    
    # Test with space format
    symptoms2 = ["severe respiratory distress", "wheezing"]
    red_flags2 = orchestrator.enrich_with_red_flags("J45.9", symptoms2)
    
    print(f"  Symptoms: {symptoms1}")
    print(f"    → Red-flags detected: {len(red_flags1)} - {red_flags1}")
    print(f"  Symptoms: {symptoms2}")
    print(f"    → Red-flags detected: {len(red_flags2)} - {red_flags2}")
    
    test3_passed = len(red_flags1) == len(red_flags2) and len(red_flags1) > 0
    print(f"  {'✅ PASS' if test3_passed else '❌ FAIL'}: Both formats detect same number of red-flags\n")
    
    # Overall result
    print("=" * 70)
    all_passed = test1_passed and test2_passed and test3_passed
    
    if all_passed:
        print("  🎉 SUCCESS: Problem statement scenario is fully resolved!")
        print("  ✅ 'severe_respiratory_distress' and 'severe respiratory distress'")
        print("     now match identically across all modules.")
    else:
        print("  ❌ FAILURE: Some tests did not pass.")
    
    print("=" * 70 + "\n")
    
    return all_passed


def test_additional_edge_cases():
    """Test additional edge cases for robustness."""
    print("Testing Additional Edge Cases")
    print("=" * 70)
    
    orchestrator = DiagnosisOrchestrator()
    
    # Edge case 1: Mixed separators in single symptom
    symptoms1 = ["severe_respiratory-distress"]  # Mixed underscore and hyphen
    red_flags1 = orchestrator.enrich_with_red_flags("J45.9", symptoms1)
    
    # Edge case 2: Leading/trailing whitespace
    symptoms2 = ["  severe respiratory distress  "]  # Extra spaces
    red_flags2 = orchestrator.enrich_with_red_flags("J45.9", symptoms2)
    
    # Edge case 3: All caps with underscores
    symptoms3 = ["SEVERE_RESPIRATORY_DISTRESS"]
    red_flags3 = orchestrator.enrich_with_red_flags("J45.9", symptoms3)
    
    # Edge case 4: Hyphens only
    symptoms4 = ["severe-respiratory-distress"]
    red_flags4 = orchestrator.enrich_with_red_flags("J45.9", symptoms4)
    
    print(f"  Mixed separators:      {symptoms1[0]:40s} → {len(red_flags1)} red-flags")
    print(f"  Extra whitespace:      {symptoms2[0]:40s} → {len(red_flags2)} red-flags")
    print(f"  All caps + underscores:{symptoms3[0]:40s} → {len(red_flags3)} red-flags")
    print(f"  Hyphens only:          {symptoms4[0]:40s} → {len(red_flags4)} red-flags")
    
    # All should detect the same number of red-flags
    counts = [len(red_flags1), len(red_flags2), len(red_flags3), len(red_flags4)]
    all_same = all(c == counts[0] for c in counts) and counts[0] > 0
    
    print(f"\n  {'✅ PASS' if all_same else '❌ FAIL'}: All edge cases produce consistent results")
    print("=" * 70 + "\n")
    
    return all_same


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  SYMPTOM NORMALIZATION FIX VERIFICATION")
    print("  Testing the exact problem from the issue description")
    print("=" * 70)
    
    # Run tests
    test1_passed = test_problem_statement_scenario()
    test2_passed = test_additional_edge_cases()
    
    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL VERIFICATION RESULTS")
    print("=" * 70)
    
    if test1_passed and test2_passed:
        print("\n  ✅ All verification tests PASSED!")
        print("\n  The symptom normalization issue has been successfully resolved.")
        print("  Symptoms with underscores, spaces, hyphens, and mixed case are")
        print("  now handled consistently across all modules:")
        print("    • DialogueEngine")
        print("    • BayesianSymptomNetwork")
        print("    • DiagnosisOrchestrator")
        print("\n" + "=" * 70 + "\n")
        return 0
    else:
        print("\n  ❌ Some verification tests FAILED.")
        print("\n" + "=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
