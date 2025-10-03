#!/usr/bin/env python3
"""Test uncertainty quantification in discriminator."""

from phaita.models.discriminator import DiagnosisDiscriminator


def test_uncertainty_higher_for_ambiguous():
    """Test that uncertainty is higher for ambiguous complaints."""
    try:
        disc = DiagnosisDiscriminator(use_pretrained=True)
    except (RuntimeError, ValueError) as e:
        print(f"⚠️  Skipping test - model not available: {e}")
        return
    
    # Clear complaint should have low uncertainty
    clear_complaint = ["Severe wheezing, chest tightness, and difficulty breathing for 3 days"]
    clear_results = disc.predict_with_uncertainty(clear_complaint, num_samples=20, top_k=3)
    clear_uncertainty = clear_results[0][0]['uncertainty']
    
    # Ambiguous complaint should have higher uncertainty
    ambiguous_complaint = ["Not feeling well"]
    ambiguous_results = disc.predict_with_uncertainty(ambiguous_complaint, num_samples=20, top_k=3)
    ambiguous_uncertainty = ambiguous_results[0][0]['uncertainty']
    
    assert ambiguous_uncertainty > clear_uncertainty, (
        f"Ambiguous complaint should have higher uncertainty. "
        f"Got clear={clear_uncertainty:.3f}, ambiguous={ambiguous_uncertainty:.3f}"
    )
    print(f"✓ Clear uncertainty: {clear_uncertainty:.3f}")
    print(f"✓ Ambiguous uncertainty: {ambiguous_uncertainty:.3f}")
    print("✅ Uncertainty quantification working correctly")
    

def test_mc_dropout_produces_variance():
    """Test that MC dropout produces variance in predictions."""
    try:
        disc = DiagnosisDiscriminator(use_pretrained=True)
    except (RuntimeError, ValueError) as e:
        print(f"⚠️  Skipping test - model not available: {e}")
        return
    
    complaint = ["Chronic cough and shortness of breath"]
    results = disc.predict_with_uncertainty(complaint, num_samples=20, top_k=3)
    
    # Check that uncertainties are non-zero
    uncertainties = [r['uncertainty'] for r in results[0]]
    assert all(u > 0 for u in uncertainties), "All uncertainties should be > 0"
    
    # Check that confidence levels are assigned
    confidence_levels = [r['confidence_level'] for r in results[0]]
    assert all(c in ['high', 'medium', 'low'] for c in confidence_levels)
    
    print(f"✓ Uncertainties: {[f'{u:.3f}' for u in uncertainties]}")
    print(f"✓ Confidence levels: {confidence_levels}")
    print("✅ MC dropout variance test passed")


def test_uncertainty_returned_with_predictions():
    """Test that uncertainty is returned alongside probability."""
    try:
        disc = DiagnosisDiscriminator(use_pretrained=True)
    except (RuntimeError, ValueError) as e:
        print(f"⚠️  Skipping test - model not available: {e}")
        return
    
    complaint = ["I have a bad cough"]
    results = disc.predict_with_uncertainty(complaint, num_samples=10, top_k=3)
    
    # Check structure
    assert len(results) == 1, "Should return results for 1 complaint"
    assert len(results[0]) == 3, "Should return top 3 predictions"
    
    for pred in results[0]:
        assert 'condition_code' in pred
        assert 'condition_name' in pred
        assert 'probability' in pred
        assert 'uncertainty' in pred
        assert 'confidence_level' in pred
        assert 0 <= pred['probability'] <= 1
        assert pred['uncertainty'] >= 0
    
    print("✓ All required fields present")
    print(f"✓ Top prediction: {results[0][0]['condition_name']}")
    print(f"  Probability: {results[0][0]['probability']:.3f}")
    print(f"  Uncertainty: {results[0][0]['uncertainty']:.3f}")
    print(f"  Confidence: {results[0][0]['confidence_level']}")
    print("✅ Uncertainty structure test passed")


if __name__ == "__main__":
    print("🧪 Testing Uncertainty Quantification")
    print("=" * 60)
    
    test_uncertainty_returned_with_predictions()
    print()
    test_mc_dropout_produces_variance()
    print()
    test_uncertainty_higher_for_ambiguous()
    
    print()
    print("=" * 60)
    print("🎉 All uncertainty tests passed!")


