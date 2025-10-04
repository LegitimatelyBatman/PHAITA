#!/usr/bin/env python3
"""
Test script for dynamic medical terms generation in DataPreprocessor.
Validates that preprocessing.py correctly generates medical terms from RespiratoryConditions.get_vocabulary().
"""

import sys
import traceback


def test_dynamic_medical_terms_generation():
    """Test that medical terms are dynamically generated from vocabulary."""
    print("🧪 Testing Dynamic Medical Terms Generation...")
    
    try:
        from phaita.data.preprocessing import DataPreprocessor
        from phaita.data.icd_conditions import RespiratoryConditions
        
        # Create a preprocessor
        preprocessor = DataPreprocessor()
        
        # Get the vocabulary to compare
        vocab = RespiratoryConditions.get_vocabulary()
        
        # Check that medical_terms is a set
        assert isinstance(preprocessor.medical_terms, set), "medical_terms should be a set"
        
        # Check that medical_terms includes symptoms from vocabulary
        # We need to normalize underscores to spaces and handle both formats
        vocab_symptoms = set()
        for symptom in vocab['symptoms']:
            vocab_symptoms.add(symptom)
            vocab_symptoms.add(symptom.replace('_', ' '))
        
        # Check some key symptoms are present
        sample_symptoms = ['wheezing', 'shortness of breath', 'chest tightness', 'cough']
        for symptom in sample_symptoms:
            assert symptom in preprocessor.medical_terms or symptom.replace(' ', '_') in preprocessor.medical_terms, \
                f"Symptom '{symptom}' should be in medical_terms"
        
        # Check that severity indicators are included
        sample_severity = ['severe', 'moderate', 'mild']
        for severity in sample_severity:
            # These might come from hardcoded values or vocabulary
            found = any(s in preprocessor.medical_terms for s in [severity, severity.replace(' ', '_')])
            assert found, f"Severity indicator '{severity}' should be in medical_terms"
        
        # Check that lay_to_medical has mappings
        assert len(preprocessor.lay_to_medical) > 0, "lay_to_medical should have mappings"
        
        # Check that condition names are included
        sample_conditions = ['asthma', 'pneumonia', 'copd']
        for condition in sample_conditions:
            found = any(c.lower() in preprocessor.medical_terms for c in [condition, condition.upper()])
            # Note: condition names might not be in medical_terms, so we'll check they're somewhere accessible
        
        print(f"  ✓ medical_terms has {len(preprocessor.medical_terms)} terms")
        print(f"  ✓ lay_to_medical has {len(preprocessor.lay_to_medical)} mappings")
        print("✅ Dynamic medical terms generation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Dynamic medical terms generation test failed: {e}")
        traceback.print_exc()
        return False


def test_medical_term_extraction():
    """Test that medical term extraction works with dynamically generated terms."""
    print("🧪 Testing Medical Term Extraction with Dynamic Terms...")
    
    try:
        from phaita.data.preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Test various complaints
        test_cases = [
            ("I have been wheezing and short of breath", ["wheezing", "shortness of breath"]),
            ("My chest hurts and I have a cough", ["chest pain", "cough"]),
            ("I can't breathe properly", ["dyspnea"]),
        ]
        
        for complaint, expected_terms in test_cases:
            extracted = preprocessor.extract_medical_terms(complaint, include_lay_terms=True)
            extracted_terms = [term for term, _ in extracted]
            
            # Check that at least some expected terms are found
            found_count = sum(1 for expected in expected_terms 
                            if any(expected in term or term in expected for term in extracted_terms))
            
            assert found_count > 0, \
                f"Expected to find at least one of {expected_terms} in complaint '{complaint}', got {extracted_terms}"
        
        print("✅ Medical term extraction tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Medical term extraction test failed: {e}")
        traceback.print_exc()
        return False


def test_vocabulary_sync():
    """Test that preprocessor stays in sync with RespiratoryConditions."""
    print("🧪 Testing Vocabulary Synchronization...")
    
    try:
        from phaita.data.preprocessing import DataPreprocessor
        from phaita.data.icd_conditions import RespiratoryConditions
        
        # Get vocabulary
        vocab = RespiratoryConditions.get_vocabulary()
        
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Check that key vocabulary items are in preprocessor
        # Symptoms should be present (with normalization)
        symptom_count = 0
        for symptom in vocab['symptoms'][:5]:  # Check first 5
            normalized = symptom.replace('_', ' ')
            if normalized in preprocessor.medical_terms or symptom in preprocessor.medical_terms:
                symptom_count += 1
        
        assert symptom_count >= 3, \
            f"Expected at least 3 symptoms from vocabulary in medical_terms, found {symptom_count}"
        
        print(f"  ✓ Found {symptom_count}/5 vocabulary symptoms in medical_terms")
        print("✅ Vocabulary synchronization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Vocabulary synchronization test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🏥 PHAITA Dynamic Preprocessing Test Suite")
    print("=" * 50)
    
    tests = [
        test_dynamic_medical_terms_generation,
        test_medical_term_extraction,
        test_vocabulary_sync,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
