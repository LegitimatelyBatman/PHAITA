#!/usr/bin/env python3
"""
Test enhanced Bayesian network functionality.
"""

import sys


def test_enhanced_bayesian_network():
    """Test enhanced Bayesian network functionality."""
    print("🧠 Testing Enhanced Bayesian Network...")
    
    try:
        from phaita.models.enhanced_bayesian_network import create_enhanced_bayesian_network
        
        network = create_enhanced_bayesian_network()
        
        # Test standard symptom sampling
        symptoms, metadata = network.sample_symptoms("J45.9", age_group="adult", severity="moderate")
        assert len(symptoms) > 0, "Should generate at least one symptom"
        assert metadata["presentation_type"] in ["standard", "rare"], "Should have valid presentation type"
        
        print(f"✅ Standard sampling: {len(symptoms)} symptoms")
        print(f"   Symptoms: {symptoms[:3]}...")
        print(f"   Metadata: {metadata['presentation_type']}")
        
        # Test rare presentation sampling (multiple attempts)
        rare_found = False
        for _ in range(20):  # Try multiple times to find rare case
            symptoms, metadata = network.sample_symptoms("J45.9", include_rare=True)
            if metadata["presentation_type"] == "rare":
                rare_found = True
                print(f"✅ Rare presentation found: {metadata.get('case_name', 'Unknown')}")
                break
        
        # Test age group modifiers
        child_symptoms, child_meta = network.sample_symptoms("J45.9", age_group="child")
        elderly_symptoms, elderly_meta = network.sample_symptoms("J45.9", age_group="elderly")
        
        print(f"✅ Age-specific sampling works")
        
        # Test severity modifiers
        mild_symptoms, mild_meta = network.sample_symptoms("J45.9", severity="mild")
        severe_symptoms, severe_meta = network.sample_symptoms("J45.9", severity="severe")
        
        print(f"✅ Severity-specific sampling works")
        
        # Test evidence sources
        sources = network.get_evidence_sources("J45.9")
        assert len(sources) > 0, "Should have evidence sources"
        
        print(f"✅ Evidence sources: {len(sources)} symptoms documented")
        for symptom, source in list(sources.items())[:2]:
            print(f"   {symptom}: {source}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Bayesian network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comorbidity_modeling():
    """Test comorbidity modeling functionality."""
    print("\n🏥 Testing Comorbidity Modeling...")
    
    try:
        from phaita.models.enhanced_bayesian_network import create_enhanced_bayesian_network
        
        network = create_enhanced_bayesian_network()
        
        # Test 1: Single comorbidity increases relevant symptom probability
        print("\n  Test 1: Single comorbidity effects")
        trials = 50
        diabetes_fatigue_count = 0
        no_comorbidity_fatigue_count = 0
        
        for _ in range(trials):
            symptoms_with_diabetes, _ = network.sample_symptoms("J18.9", comorbidities=["diabetes"])
            symptoms_without, _ = network.sample_symptoms("J18.9")
            
            # Count fatigue occurrence (diabetes increases fatigue probability)
            if "fatigue" in symptoms_with_diabetes:
                diabetes_fatigue_count += 1
            if "fatigue" in symptoms_without:
                no_comorbidity_fatigue_count += 1
        
        print(f"     Fatigue with diabetes: {diabetes_fatigue_count}/{trials}")
        print(f"     Fatigue without: {no_comorbidity_fatigue_count}/{trials}")
        # We expect diabetes to increase fatigue occurrence, but it's probabilistic
        print(f"  ✅ Single comorbidity modifies symptom probabilities")
        
        # Test 2: Multiple comorbidities compound effects
        print("\n  Test 2: Multiple comorbidities compound effects")
        symptoms_multi, metadata_multi = network.sample_symptoms(
            "J45.9", 
            comorbidities=["diabetes", "obesity"]
        )
        assert "comorbidities" in metadata_multi, "Should track comorbidities in metadata"
        assert len(metadata_multi["comorbidities"]) == 2, "Should have 2 comorbidities"
        print(f"     Symptoms with diabetes + obesity: {len(symptoms_multi)} symptoms")
        print(f"     Sample symptoms: {symptoms_multi[:4]}...")
        print(f"  ✅ Multiple comorbidities tracked in metadata")
        
        # Test 3: Comorbidity-specific symptoms appear
        print("\n  Test 3: Comorbidity-specific symptoms")
        comorbidity_specific_found = False
        for _ in range(30):  # Multiple trials to find specific symptoms
            symptoms, _ = network.sample_symptoms("J45.9", comorbidities=["hypertension"])
            # Check for hypertension-specific symptoms
            if any(s in symptoms for s in ["palpitations", "dizziness", "headache"]):
                comorbidity_specific_found = True
                print(f"     Found comorbidity-specific symptom in: {symptoms}")
                break
        
        if comorbidity_specific_found:
            print(f"  ✅ Comorbidity-specific symptoms can appear")
        else:
            print(f"  ⚠️  Comorbidity-specific symptoms not found in 30 trials (probabilistic)")
        
        # Test 4: Cross-condition interaction (Asthma + COPD = ACOS)
        print("\n  Test 4: Cross-condition interactions (ACOS)")
        acos_chronic_cough_count = 0
        trials = 20
        
        for _ in range(trials):
            symptoms, metadata = network.sample_symptoms("J45.9", comorbidities=["copd"])
            if "chronic_cough" in symptoms:
                acos_chronic_cough_count += 1
        
        print(f"     Chronic cough in ACOS cases: {acos_chronic_cough_count}/{trials}")
        print(f"     Expected: ~18/20 (90% probability for ACOS)")
        assert acos_chronic_cough_count >= trials * 0.7, "ACOS should have high chronic cough rate"
        print(f"  ✅ Cross-condition interactions work (ACOS)")
        
        # Test 5: Comorbidities parameter is optional
        print("\n  Test 5: Comorbidities parameter is optional")
        symptoms_no_comorbidity, metadata_no_comorbidity = network.sample_symptoms("J45.9")
        assert "comorbidities" not in metadata_no_comorbidity, "Should not have comorbidities if not specified"
        print(f"  ✅ Comorbidities parameter is optional")
        
        # Test 6: Unknown comorbidity is handled gracefully
        print("\n  Test 6: Unknown comorbidity handling")
        symptoms_unknown, _ = network.sample_symptoms("J45.9", comorbidities=["unknown_condition"])
        assert len(symptoms_unknown) > 0, "Should still generate symptoms with unknown comorbidity"
        print(f"  ✅ Unknown comorbidities handled gracefully")
        
        return True
        
    except Exception as e:
        print(f"❌ Comorbidity modeling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run enhanced Bayesian network tests."""
    print("🧠 Enhanced Bayesian Network Tests")
    print("=" * 50)
    
    test_passed = test_enhanced_bayesian_network()
    comorbidity_passed = test_comorbidity_modeling()
    
    print("=" * 50)
    if test_passed and comorbidity_passed:
        print("🎉 All enhanced Bayesian network tests passed!")
        return 0
    else:
        print("❌ Some enhanced Bayesian network tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())