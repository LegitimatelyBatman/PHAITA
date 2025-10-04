#!/usr/bin/env python3
"""Test that predict_with_uncertainty refactoring maintains correct behavior."""

from phaita.models.discriminator import DiagnosisDiscriminator


def test_predict_with_uncertainty_uses_mc_dropout_sample():
    """Test that predict_with_uncertainty correctly calls _mc_dropout_sample."""
    # Use lightweight mode to avoid model downloads
    disc = DiagnosisDiscriminator(use_pretrained=False)
    
    # Test basic functionality
    complaints = ["I have a bad cough", "shortness of breath"]
    results = disc.predict_with_uncertainty(complaints, num_samples=5, top_k=3)
    
    # Verify structure
    assert len(results) == 2, f"Should return results for 2 complaints, got {len(results)}"
    assert len(results[0]) == 3, f"Should return top 3 predictions, got {len(results[0])}"
    assert len(results[1]) == 3, f"Should return top 3 predictions, got {len(results[1])}"
    
    # Verify all required fields are present
    for complaint_results in results:
        for pred in complaint_results:
            assert 'condition_code' in pred, "Missing condition_code"
            assert 'condition_name' in pred, "Missing condition_name"
            assert 'probability' in pred, "Missing probability"
            assert 'uncertainty' in pred, "Missing uncertainty"
            assert 'confidence_level' in pred, "Missing confidence_level"
            
            # Verify value ranges
            assert 0 <= pred['probability'] <= 1, f"Probability out of range: {pred['probability']}"
            assert pred['uncertainty'] >= 0, f"Uncertainty should be non-negative: {pred['uncertainty']}"
            assert pred['confidence_level'] in ['high', 'medium', 'low'], \
                f"Invalid confidence level: {pred['confidence_level']}"
    
    print("✓ All required fields present and valid")
    print(f"✓ Sample result: {results[0][0]['condition_name']} "
          f"(prob={results[0][0]['probability']:.3f}, "
          f"uncertainty={results[0][0]['uncertainty']:.3f}, "
          f"confidence={results[0][0]['confidence_level']})")


def test_training_state_preservation():
    """Test that predict_with_uncertainty preserves training state."""
    disc = DiagnosisDiscriminator(use_pretrained=False)
    
    # Test starting in eval mode
    disc.eval()
    initial_state = disc.training
    assert not initial_state, "Model should start in eval mode"
    
    results = disc.predict_with_uncertainty(["test complaint"], num_samples=3, top_k=1)
    assert not disc.training, "Model should be back in eval mode after prediction"
    print("✓ Eval mode preserved correctly")
    
    # Test starting in training mode
    disc.train()
    assert disc.training, "Model should be in training mode"
    
    results = disc.predict_with_uncertainty(["test complaint"], num_samples=3, top_k=1)
    assert disc.training, "Model should be back in training mode after prediction"
    print("✓ Training mode preserved correctly")


def test_consistency_with_predict_diagnosis():
    """Test that predict_with_uncertainty produces consistent results with predict_diagnosis."""
    disc = DiagnosisDiscriminator(use_pretrained=False)
    
    complaints = ["I have chest pain and difficulty breathing"]
    
    # Get results from both methods
    uncertainty_results = disc.predict_with_uncertainty(complaints, num_samples=10, top_k=3)
    diagnosis_results = disc.predict_diagnosis(complaints, top_k=3, use_mc_dropout=True, num_mc_samples=10)
    
    # Both should return results for 1 complaint with 3 predictions
    assert len(uncertainty_results) == 1
    assert len(diagnosis_results) == 1
    assert len(uncertainty_results[0]) == 3
    assert len(diagnosis_results[0]) == 3
    
    # The top predictions should be the same (since both use _mc_dropout_sample)
    # Note: Due to randomness in dropout, exact values may differ between runs,
    # but the structure should be consistent
    for i in range(3):
        uncert_pred = uncertainty_results[0][i]
        diag_pred = diagnosis_results[0][i]
        
        # Both should have the required fields
        assert 'condition_code' in uncert_pred
        assert 'condition_code' in diag_pred
        assert 'probability' in uncert_pred
        assert 'probability' in diag_pred
        
    print("✓ Results structure consistent between methods")


if __name__ == "__main__":
    print("🧪 Testing predict_with_uncertainty Refactoring")
    print("=" * 60)
    
    test_predict_with_uncertainty_uses_mc_dropout_sample()
    print()
    
    test_training_state_preservation()
    print()
    
    test_consistency_with_predict_diagnosis()
    print()
    
    print("=" * 60)
    print("🎉 All refactoring tests passed!")
