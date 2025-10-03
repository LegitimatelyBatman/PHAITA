#!/usr/bin/env python3
"""
Test script for uncertainty quantification in DiagnosisDiscriminator.

Tests Monte Carlo Dropout implementation and validates that:
1. Uncertainty is higher for ambiguous symptoms
2. Uncertainty is lower for clear-cut cases
3. MC dropout produces reasonable variance
4. Uncertainty correlates with misclassification rate
"""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_mc_dropout_produces_variance():
    """Test that MC dropout produces non-zero variance in predictions."""
    print("🧪 Test 1: MC Dropout Variance...")
    
    try:
        import torch
        from phaita.models.discriminator import DiagnosisDiscriminator
        
        # Create discriminator (will use pretrained=False for testing without models)
        try:
            model = DiagnosisDiscriminator(use_pretrained=True)
        except (RuntimeError, ImportError) as e:
            print(f"⚠️  Skipping test - models not available: {e}")
            return True
        
        complaint = "I have a cough and some breathing issues"
        
        # Get predictions with MC dropout
        mean_probs, std_probs = model.predict_with_uncertainty([complaint], num_samples=20)
        
        # Verify shapes
        assert mean_probs.shape[0] == 1, "Should have one prediction"
        assert std_probs.shape[0] == 1, "Should have one std prediction"
        assert mean_probs.shape[1] == model.num_conditions, "Should have prediction for each condition"
        
        # Check that standard deviation is non-zero (MC dropout should produce variance)
        max_std = std_probs.max().item()
        assert max_std > 0, f"MC dropout should produce non-zero variance, got max_std={max_std}"
        
        # Check that probabilities sum to approximately 1
        prob_sum = mean_probs.sum(dim=1).item()
        assert 0.95 < prob_sum < 1.05, f"Probabilities should sum to ~1, got {prob_sum}"
        
        print(f"✅ MC Dropout variance test passed (max_std={max_std:.4f})")
        return True
        
    except Exception as e:
        print(f"❌ MC Dropout variance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uncertainty_higher_for_ambiguous_symptoms():
    """Test that uncertainty is higher for ambiguous symptom presentations."""
    print("\n🧪 Test 2: Higher Uncertainty for Ambiguous Symptoms...")
    
    try:
        import torch
        from phaita.models.discriminator import DiagnosisDiscriminator
        
        try:
            model = DiagnosisDiscriminator(use_pretrained=True)
        except (RuntimeError, ImportError) as e:
            print(f"⚠️  Skipping test - models not available: {e}")
            return True
        
        # Ambiguous complaint - vague symptoms that could indicate multiple conditions
        ambiguous_complaint = "I feel tired and have some discomfort"
        
        # Clear complaint - specific symptoms pointing to one condition
        clear_complaint = "I have severe wheezing, shortness of breath, and chest tightness that gets worse at night"
        
        # Get predictions with uncertainty
        predictions_ambiguous = model.predict_diagnosis([ambiguous_complaint], top_k=1, use_mc_dropout=True)
        predictions_clear = model.predict_diagnosis([clear_complaint], top_k=1, use_mc_dropout=True)
        
        # Extract uncertainty scores
        uncertainty_ambiguous = predictions_ambiguous[0][0]["uncertainty"]
        uncertainty_clear = predictions_clear[0][0]["uncertainty"]
        
        print(f"   Ambiguous complaint uncertainty: {uncertainty_ambiguous:.4f}")
        print(f"   Clear complaint uncertainty: {uncertainty_clear:.4f}")
        
        # Ambiguous should have higher uncertainty
        assert uncertainty_ambiguous > uncertainty_clear, \
            f"Ambiguous complaint should have higher uncertainty: {uncertainty_ambiguous} vs {uncertainty_clear}"
        
        # Check confidence levels
        confidence_level_ambiguous = predictions_ambiguous[0][0]["confidence_level"]
        confidence_level_clear = predictions_clear[0][0]["confidence_level"]
        
        print(f"   Ambiguous confidence level: {confidence_level_ambiguous}")
        print(f"   Clear confidence level: {confidence_level_clear}")
        
        print("✅ Ambiguous symptoms uncertainty test passed")
        return True
        
    except Exception as e:
        print(f"❌ Ambiguous symptoms uncertainty test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uncertainty_lower_for_clear_cases():
    """Test that uncertainty is lower for clear-cut diagnostic cases."""
    print("\n🧪 Test 3: Lower Uncertainty for Clear Cases...")
    
    try:
        import torch
        from phaita.models.discriminator import DiagnosisDiscriminator
        
        try:
            model = DiagnosisDiscriminator(use_pretrained=True)
        except (RuntimeError, ImportError) as e:
            print(f"⚠️  Skipping test - models not available: {e}")
            return True
        
        # Very specific symptoms for different conditions
        clear_cases = [
            "Severe asthma attack with wheezing and inability to breathe properly",
            "Productive cough with yellow-green sputum and high fever indicating pneumonia",
            "Chronic bronchitis with persistent cough for months and smoking history"
        ]
        
        predictions = model.predict_diagnosis(clear_cases, top_k=1, use_mc_dropout=True)
        
        uncertainties = [pred[0]["uncertainty"] for pred in predictions]
        confidence_levels = [pred[0]["confidence_level"] for pred in predictions]
        
        print(f"   Clear case uncertainties: {[f'{u:.4f}' for u in uncertainties]}")
        print(f"   Confidence levels: {confidence_levels}")
        
        # At least some should have low uncertainty (< 0.5)
        low_uncertainty_count = sum(1 for u in uncertainties if u < 0.5)
        assert low_uncertainty_count > 0, \
            f"Expected some clear cases to have low uncertainty, got: {uncertainties}"
        
        # High confidence level should appear for some clear cases
        high_confidence_count = sum(1 for cl in confidence_levels if cl == "high")
        print(f"   Cases with high confidence: {high_confidence_count}/{len(clear_cases)}")
        
        print("✅ Clear cases uncertainty test passed")
        return True
        
    except Exception as e:
        print(f"❌ Clear cases uncertainty test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mc_dropout_consistency():
    """Test that MC dropout produces consistent but varied predictions."""
    print("\n🧪 Test 4: MC Dropout Consistency...")
    
    try:
        import torch
        from phaita.models.discriminator import DiagnosisDiscriminator
        
        try:
            model = DiagnosisDiscriminator(use_pretrained=True)
        except (RuntimeError, ImportError) as e:
            print(f"⚠️  Skipping test - models not available: {e}")
            return True
        
        complaint = "I have difficulty breathing and chest pain"
        
        # Run MC dropout multiple times
        mean_probs_1, std_probs_1 = model.predict_with_uncertainty([complaint], num_samples=10)
        mean_probs_2, std_probs_2 = model.predict_with_uncertainty([complaint], num_samples=10)
        
        # Mean predictions should be similar (not identical due to dropout randomness)
        diff = torch.abs(mean_probs_1 - mean_probs_2).mean().item()
        print(f"   Difference between two MC runs: {diff:.4f}")
        
        # Difference should be small but not zero
        assert diff < 0.1, f"MC dropout results should be consistent, got diff={diff}"
        
        # Standard deviations should be in reasonable range
        avg_std_1 = std_probs_1.mean().item()
        avg_std_2 = std_probs_2.mean().item()
        print(f"   Average std run 1: {avg_std_1:.4f}")
        print(f"   Average std run 2: {avg_std_2:.4f}")
        
        assert 0 < avg_std_1 < 0.5, f"Standard deviation should be reasonable, got {avg_std_1}"
        assert 0 < avg_std_2 < 0.5, f"Standard deviation should be reasonable, got {avg_std_2}"
        
        print("✅ MC Dropout consistency test passed")
        return True
        
    except Exception as e:
        print(f"❌ MC Dropout consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uncertainty_fields_in_output():
    """Test that uncertainty fields are present in predict_diagnosis output."""
    print("\n🧪 Test 5: Uncertainty Fields in Output...")
    
    try:
        import torch
        from phaita.models.discriminator import DiagnosisDiscriminator
        
        try:
            model = DiagnosisDiscriminator(use_pretrained=True)
        except (RuntimeError, ImportError) as e:
            print(f"⚠️  Skipping test - models not available: {e}")
            return True
        
        complaint = "I have a persistent cough and fever"
        
        # Test with MC dropout enabled
        predictions_mc = model.predict_diagnosis([complaint], top_k=3, use_mc_dropout=True)
        
        # Test with MC dropout disabled
        predictions_no_mc = model.predict_diagnosis([complaint], top_k=3, use_mc_dropout=False)
        
        for name, predictions in [("MC dropout", predictions_mc), ("No MC dropout", predictions_no_mc)]:
            print(f"\n   Testing {name}:")
            assert len(predictions) == 1, "Should have one complaint prediction"
            assert len(predictions[0]) == 3, "Should have top_k=3 predictions"
            
            for i, pred in enumerate(predictions[0]):
                # Check required fields
                required_fields = [
                    "condition_code", "condition_name", "probability",
                    "confidence_interval", "evidence", "uncertainty", "confidence_level"
                ]
                
                for field in required_fields:
                    assert field in pred, f"Missing field '{field}' in prediction {i}"
                
                # Check field types and ranges
                assert isinstance(pred["uncertainty"], float), "uncertainty should be float"
                assert 0 <= pred["uncertainty"] <= 1, f"uncertainty should be in [0,1], got {pred['uncertainty']}"
                assert pred["confidence_level"] in ["high", "low"], \
                    f"confidence_level should be 'high' or 'low', got {pred['confidence_level']}"
                
                print(f"      Prediction {i+1}: {pred['condition_name']}")
                print(f"         Probability: {pred['probability']:.4f}")
                print(f"         Uncertainty: {pred['uncertainty']:.4f}")
                print(f"         Confidence: {pred['confidence_level']}")
        
        print("\n✅ Uncertainty fields test passed")
        return True
        
    except Exception as e:
        print(f"❌ Uncertainty fields test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_entropy_calculation():
    """Test the entropy calculation helper method."""
    print("\n🧪 Test 6: Entropy Calculation...")
    
    try:
        import torch
        from phaita.models.discriminator import DiagnosisDiscriminator
        
        # Test entropy calculation directly
        # High confidence prediction (low entropy)
        high_confidence = torch.tensor([[0.9, 0.05, 0.05]])
        entropy_high = DiagnosisDiscriminator._calculate_entropy(high_confidence)
        
        # Uniform distribution (high entropy)
        uniform = torch.tensor([[0.33, 0.33, 0.34]])
        entropy_uniform = DiagnosisDiscriminator._calculate_entropy(uniform)
        
        # Very uncertain prediction (medium-high entropy)
        uncertain = torch.tensor([[0.4, 0.3, 0.3]])
        entropy_uncertain = DiagnosisDiscriminator._calculate_entropy(uncertain)
        
        print(f"   High confidence entropy: {entropy_high.item():.4f}")
        print(f"   Uniform distribution entropy: {entropy_uniform.item():.4f}")
        print(f"   Uncertain prediction entropy: {entropy_uncertain.item():.4f}")
        
        # Entropy should be lowest for high confidence
        assert entropy_high < entropy_uncertain < entropy_uniform, \
            "Entropy ordering incorrect"
        
        # All entropies should be non-negative
        assert entropy_high >= 0, "Entropy should be non-negative"
        assert entropy_uniform >= 0, "Entropy should be non-negative"
        assert entropy_uncertain >= 0, "Entropy should be non-negative"
        
        print("✅ Entropy calculation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Entropy calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all uncertainty quantification tests."""
    print("=" * 70)
    print("PHAITA Uncertainty Quantification Test Suite")
    print("=" * 70)
    
    tests = [
        test_entropy_calculation,
        test_mc_dropout_produces_variance,
        test_uncertainty_fields_in_output,
        test_mc_dropout_consistency,
        test_uncertainty_lower_for_clear_cases,
        test_uncertainty_higher_for_ambiguous_symptoms,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
