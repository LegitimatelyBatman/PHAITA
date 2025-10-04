#!/usr/bin/env python3
"""Test script for text utility functions, especially symptom normalization."""

import sys
import traceback


def test_normalize_symptom():
    """Test the normalize_symptom function."""
    print("🧪 Testing normalize_symptom()...")
    
    try:
        from phaita.utils.text import normalize_symptom
        
        # Test basic normalization
        assert normalize_symptom('Shortness_of_Breath') == 'shortness of breath'
        assert normalize_symptom('chest-pain') == 'chest pain'
        assert normalize_symptom('FEVER') == 'fever'
        
        # Test mixed separators
        assert normalize_symptom('Severe_Respiratory-Distress') == 'severe respiratory distress'
        
        # Test whitespace handling
        assert normalize_symptom('  wheezing  ') == 'wheezing'
        assert normalize_symptom('\tcough\n') == 'cough'
        
        # Test multiple separators
        assert normalize_symptom('chest___pain') == 'chest pain'
        assert normalize_symptom('chest---pain') == 'chest pain'
        
        # Test already normalized
        assert normalize_symptom('shortness of breath') == 'shortness of breath'
        
        # Test empty string
        assert normalize_symptom('') == ''
        
        print("✅ normalize_symptom() tests passed")
        return True
        
    except Exception as e:
        print(f"❌ normalize_symptom() test failed: {e}")
        traceback.print_exc()
        return False


def test_normalize_symptom_to_underscores():
    """Test the normalize_symptom_to_underscores function."""
    print("🧪 Testing normalize_symptom_to_underscores()...")
    
    try:
        from phaita.utils.text import normalize_symptom_to_underscores
        
        # Test basic normalization
        assert normalize_symptom_to_underscores('Shortness of Breath') == 'shortness_of_breath'
        assert normalize_symptom_to_underscores('chest-pain') == 'chest_pain'
        assert normalize_symptom_to_underscores('FEVER') == 'fever'
        
        # Test mixed separators
        assert normalize_symptom_to_underscores('Severe Respiratory-Distress') == 'severe_respiratory_distress'
        
        # Test whitespace handling
        assert normalize_symptom_to_underscores('  wheezing  ') == 'wheezing'
        assert normalize_symptom_to_underscores('\tcough\n') == 'cough'
        
        # Test multiple separators
        assert normalize_symptom_to_underscores('chest   pain') == 'chest_pain'
        assert normalize_symptom_to_underscores('chest---pain') == 'chest_pain'
        
        # Test already normalized
        assert normalize_symptom_to_underscores('shortness_of_breath') == 'shortness_of_breath'
        
        # Test empty string
        assert normalize_symptom_to_underscores('') == ''
        
        print("✅ normalize_symptom_to_underscores() tests passed")
        return True
        
    except Exception as e:
        print(f"❌ normalize_symptom_to_underscores() test failed: {e}")
        traceback.print_exc()
        return False


def test_normalization_consistency():
    """Test that normalization functions are inverses when appropriate."""
    print("🧪 Testing normalization consistency...")
    
    try:
        from phaita.utils.text import normalize_symptom, normalize_symptom_to_underscores
        
        # Test that converting back and forth works
        symptom_space = 'shortness of breath'
        symptom_underscore = 'shortness_of_breath'
        
        # Spaces -> underscores -> spaces
        assert normalize_symptom(normalize_symptom_to_underscores(symptom_space)) == symptom_space
        
        # Underscores -> spaces -> underscores (note: not perfect inverse due to multiple spaces)
        result = normalize_symptom_to_underscores(normalize_symptom(symptom_underscore))
        assert result == symptom_underscore
        
        # Test various formats all normalize to the same space format
        variants = [
            'Shortness_of_Breath',
            'shortness-of-breath',
            'SHORTNESS OF BREATH',
            '  shortness_of_breath  ',
        ]
        
        normalized = [normalize_symptom(v) for v in variants]
        assert all(n == 'shortness of breath' for n in normalized)
        
        print("✅ Normalization consistency tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Normalization consistency test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all text utility tests."""
    print("📝 Text Utility Tests")
    print("=" * 50)
    
    tests = [
        test_normalize_symptom,
        test_normalize_symptom_to_underscores,
        test_normalization_consistency,
    ]
    
    results = [test() for test in tests]
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if all(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
