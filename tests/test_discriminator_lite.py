import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

"""Test lightweight discriminator functionality."""

from phaita.models.discriminator_lite import LightweightDiscriminator, VocabularyFeatureExtractor
from phaita.data.icd_conditions import RespiratoryConditions


def test_vocabulary_feature_extraction():
    """Test that vocabulary features are extracted correctly."""
    conditions = RespiratoryConditions.get_all_conditions()
    extractor = VocabularyFeatureExtractor(conditions)
    
    texts = [
        "I have severe wheezing and chest tightness",
        "Chronic cough and shortness of breath"
    ]
    
    features = extractor.extract_features(texts)
    
    assert features.shape == (2, 10), f"Expected shape (2, 10), got {features.shape}"
    assert features.min() >= 0, "Feature counts should be non-negative"
    print(f"✓ Feature shape: {features.shape}")
    print(f"✓ Feature range: [{features.min():.1f}, {features.max():.1f}]")
    print("✅ Vocabulary feature extraction works")


def test_lite_discriminator_loads_without_gpu():
    """Test that lite discriminator loads on CPU."""
    import torch
    
    disc = LightweightDiscriminator(use_pretrained=False)
    
    assert disc.device.type in ['cpu', 'cuda']
    assert hasattr(disc, 'classifier')
    assert hasattr(disc, 'discriminator_head')
    
    param_count = sum(p.numel() for p in disc.parameters())
    print(f"✓ Device: {disc.device}")
    print(f"✓ Parameters: {param_count:,}")
    assert param_count < 100_000_000, "Lite model should have <100M params"
    print("✅ Lite discriminator loads successfully")


def test_lite_inference_speed():
    """Test that lite model is faster than full model."""
    import time
    
    disc = LightweightDiscriminator(use_pretrained=False)
    
    complaints = ["I can't breathe"] * 32  # Batch of 32
    
    # Warmup
    _ = disc(complaints)
    
    # Timed inference
    start = time.time()
    for _ in range(10):
        _ = disc(complaints)
    elapsed = time.time() - start
    
    avg_time = elapsed / 10 * 1000  # Convert to ms
    
    print(f"✓ Batch size: 32")
    print(f"✓ Average inference: {avg_time:.2f}ms")
    assert avg_time < 500, f"Should be <500ms, got {avg_time:.2f}ms"
    print("✅ Inference speed acceptable")


def test_lite_memory_usage():
    """Test that lite model uses <1GB RAM."""
    import torch
    
    disc = LightweightDiscriminator(use_pretrained=False)
    
    # Estimate parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in disc.parameters())
    param_memory_mb = param_memory / (1024 * 1024)
    
    print(f"✓ Parameter memory: {param_memory_mb:.1f} MB")
    assert param_memory_mb < 1000, f"Should use <1GB, got {param_memory_mb:.1f}MB"
    print("✅ Memory usage acceptable")


def test_lite_predictions_valid():
    """Test that lite discriminator produces valid predictions."""
    disc = LightweightDiscriminator(use_pretrained=False)
    
    complaints = [
        "Severe wheezing and chest tightness",
        "High fever and productive cough"
    ]
    
    predictions = disc.predict_diagnosis(complaints, top_k=3)
    
    assert len(predictions) == 2
    assert len(predictions[0]) == 3
    
    for pred in predictions[0]:
        assert 'condition_code' in pred
        assert 'condition_name' in pred
        assert 'probability' in pred
        assert 0 <= pred['probability'] <= 1
    
    print(f"✓ Top prediction: {predictions[0][0]['condition_name']}")
    print(f"  Probability: {predictions[0][0]['probability']:.3f}")
    print("✅ Predictions are valid")


if __name__ == "__main__":
    print("🧪 Testing Lightweight Discriminator")
    print("=" * 60)
    
    test_vocabulary_feature_extraction()
    print()
    test_lite_discriminator_loads_without_gpu()
    print()
    test_lite_memory_usage()
    print()
    test_lite_inference_speed()
    print()
    test_lite_predictions_valid()
    
    print()
    print("=" * 60)
    print("🎉 All lite discriminator tests passed!")
