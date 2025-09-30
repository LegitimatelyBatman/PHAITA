#!/usr/bin/env python3
"""
Test enhanced Bayesian network functionality.
"""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

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

def main():
    """Run enhanced Bayesian network tests."""
    print("🧠 Enhanced Bayesian Network Tests")
    print("=" * 50)
    
    test_passed = test_enhanced_bayesian_network()
    
    print("=" * 50)
    if test_passed:
        print("🎉 Enhanced Bayesian network tests passed!")
        return 0
    else:
        print("❌ Enhanced Bayesian network tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())