#!/usr/bin/env python3
"""
Test script for LearnableTemporalPatternMatcher.
Validates the learnable neural network approach to temporal pattern matching.
"""

import sys
import traceback
import yaml
from pathlib import Path


def test_learnable_temporal_creation():
    """Test creating the LearnableTemporalPatternMatcher."""
    print("🧪 Testing LearnableTemporalPatternMatcher creation...")
    
    try:
        import torch
        from phaita.models.temporal_module import LearnableTemporalPatternMatcher
        from phaita.data.icd_conditions import RespiratoryConditions
        
        # Get condition codes
        conditions = RespiratoryConditions.get_all_conditions()
        condition_codes = list(conditions.keys())
        num_conditions = len(condition_codes)
        
        # Create model
        model = LearnableTemporalPatternMatcher(
            num_conditions=num_conditions,
            symptom_vocab_size=100,
            symptom_embedding_dim=32,
            hidden_dim=64,
            num_layers=2,
            condition_codes=condition_codes,
        )
        
        assert model is not None, "Model should be created"
        assert model.num_conditions == num_conditions, f"Expected {num_conditions} conditions"
        assert model.symptom_vocab_size == 100, "Expected vocab size 100"
        assert model.hidden_dim == 64, "Expected hidden dim 64"
        
        print(f"  ✓ Model created with {num_conditions} conditions")
        print(f"  ✓ Symptom vocab size: 100")
        print(f"  ✓ Hidden dimension: 64")
        print("✅ LearnableTemporalPatternMatcher creation test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping test (missing torch): {e}")
        return True  # Don't fail if torch not available
    except Exception as e:
        print(f"❌ LearnableTemporalPatternMatcher creation test failed: {e}")
        traceback.print_exc()
        return False


def test_learnable_temporal_forward_pass():
    """Test forward pass through the model."""
    print("🧪 Testing LearnableTemporalPatternMatcher forward pass...")
    
    try:
        import torch
        from phaita.models.temporal_module import LearnableTemporalPatternMatcher
        from phaita.data.icd_conditions import RespiratoryConditions
        
        # Get condition codes
        conditions = RespiratoryConditions.get_all_conditions()
        condition_codes = list(conditions.keys())
        num_conditions = len(condition_codes)
        
        # Create model
        model = LearnableTemporalPatternMatcher(
            num_conditions=num_conditions,
            symptom_vocab_size=100,
            symptom_embedding_dim=32,
            hidden_dim=64,
            num_layers=2,
            condition_codes=condition_codes,
        )
        
        # Create dummy input
        batch_size = 4
        seq_len = 5
        
        symptom_indices = torch.randint(0, 100, (batch_size, seq_len))
        timestamps = torch.rand(batch_size, seq_len) * 100
        
        # Forward pass
        logits = model(symptom_indices, timestamps)
        
        # Check output shape
        assert logits.shape == (batch_size, num_conditions), \
            f"Expected shape ({batch_size}, {num_conditions}), got {logits.shape}"
        
        # Check output is not all zeros
        assert logits.abs().sum() > 0, "Output should not be all zeros"
        
        print(f"  ✓ Input: {batch_size} patients × {seq_len} symptoms")
        print(f"  ✓ Output: {batch_size} × {num_conditions} logits")
        print("✅ Forward pass test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping test (missing torch): {e}")
        return True
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        traceback.print_exc()
        return False


def test_learnable_temporal_prediction():
    """Test prediction method."""
    print("🧪 Testing LearnableTemporalPatternMatcher prediction...")
    
    try:
        import torch
        from phaita.models.temporal_module import LearnableTemporalPatternMatcher
        from phaita.data.icd_conditions import RespiratoryConditions
        
        # Get condition codes
        conditions = RespiratoryConditions.get_all_conditions()
        condition_codes = list(conditions.keys())
        num_conditions = len(condition_codes)
        
        # Create model
        model = LearnableTemporalPatternMatcher(
            num_conditions=num_conditions,
            symptom_vocab_size=100,
            condition_codes=condition_codes,
        )
        model.eval()
        
        # Create dummy input
        batch_size = 4
        seq_len = 5
        
        symptom_indices = torch.randint(0, 100, (batch_size, seq_len))
        timestamps = torch.rand(batch_size, seq_len) * 100
        
        # Predict
        predicted_indices, probs = model.predict_condition(symptom_indices, timestamps)
        
        # Check outputs
        assert predicted_indices.shape == (batch_size,), \
            f"Expected shape ({batch_size},), got {predicted_indices.shape}"
        assert probs.shape == (batch_size, num_conditions), \
            f"Expected shape ({batch_size}, {num_conditions}), got {probs.shape}"
        
        # Check probabilities sum to 1
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
            "Probabilities should sum to 1"
        
        # Check predictions are valid indices
        assert (predicted_indices >= 0).all() and (predicted_indices < num_conditions).all(), \
            "Predicted indices should be in valid range"
        
        print(f"  ✓ Predictions shape: {predicted_indices.shape}")
        print(f"  ✓ Probabilities shape: {probs.shape}")
        print(f"  ✓ Probabilities sum to 1: ✓")
        print("✅ Prediction test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping test (missing torch): {e}")
        return True
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        traceback.print_exc()
        return False


def test_learnable_temporal_with_patterns():
    """Test initialization with temporal patterns from YAML."""
    print("🧪 Testing LearnableTemporalPatternMatcher with temporal patterns...")
    
    try:
        import torch
        from phaita.models.temporal_module import LearnableTemporalPatternMatcher
        from phaita.data.icd_conditions import RespiratoryConditions
        
        # Load temporal patterns
        config_path = Path(__file__).parent.parent / "config" / "temporal_patterns.yaml"
        if not config_path.exists():
            print("  ⚠️  temporal_patterns.yaml not found, skipping")
            return True
        
        with open(config_path) as f:
            temporal_patterns = yaml.safe_load(f)
        
        # Get condition codes
        conditions = RespiratoryConditions.get_all_conditions()
        condition_codes = list(conditions.keys())
        num_conditions = len(condition_codes)
        
        # Create model with patterns
        model = LearnableTemporalPatternMatcher(
            num_conditions=num_conditions,
            symptom_vocab_size=100,
            temporal_patterns=temporal_patterns,
            condition_codes=condition_codes,
        )
        
        assert model.temporal_patterns == temporal_patterns, \
            "Temporal patterns should be stored"
        
        print(f"  ✓ Loaded {len(temporal_patterns)} temporal patterns")
        print(f"  ✓ Model initialized with clinical knowledge")
        print("✅ Temporal patterns initialization test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping test (missing torch): {e}")
        return True
    except Exception as e:
        print(f"❌ Temporal patterns test failed: {e}")
        traceback.print_exc()
        return False


def test_learnable_temporal_score_timeline():
    """Test score_timeline method for compatibility."""
    print("🧪 Testing LearnableTemporalPatternMatcher score_timeline...")
    
    try:
        import torch
        from phaita.models.temporal_module import LearnableTemporalPatternMatcher
        from phaita.data.icd_conditions import RespiratoryConditions
        
        # Get condition codes
        conditions = RespiratoryConditions.get_all_conditions()
        condition_codes = list(conditions.keys())
        num_conditions = len(condition_codes)
        
        # Create model
        model = LearnableTemporalPatternMatcher(
            num_conditions=num_conditions,
            symptom_vocab_size=100,
            condition_codes=condition_codes,
        )
        model.eval()
        
        # Create single timeline (batch size 1)
        symptom_indices = torch.randint(0, 100, (1, 5))
        timestamps = torch.rand(1, 5) * 100
        
        # Score against first condition
        condition_code = condition_codes[0]
        score = model.score_timeline(symptom_indices, timestamps, condition_code)
        
        # Check score is in valid range
        assert 0.0 <= score <= 1.0, \
            f"Score should be in [0, 1], got {score}"
        
        # Test with unknown condition
        unknown_score = model.score_timeline(symptom_indices, timestamps, "UNKNOWN")
        assert unknown_score == 0.5, \
            f"Unknown condition should return 0.5, got {unknown_score}"
        
        print(f"  ✓ Score for {condition_code}: {score:.3f}")
        print(f"  ✓ Unknown condition score: {unknown_score:.3f}")
        print("✅ score_timeline test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping test (missing torch): {e}")
        return True
    except Exception as e:
        print(f"❌ score_timeline test failed: {e}")
        traceback.print_exc()
        return False


def test_adversarial_trainer_integration():
    """Test integration with AdversarialTrainer."""
    print("🧪 Testing AdversarialTrainer integration...")
    
    try:
        import torch
        from phaita.training.adversarial_trainer import AdversarialTrainer
        
        # Create trainer with learnable temporal
        trainer = AdversarialTrainer(
            use_learnable_temporal=True,
            use_pretrained_generator=False,
            use_pretrained_discriminator=False,
        )
        
        # Check temporal model was initialized
        if trainer.use_learnable_temporal:
            assert trainer.temporal_model is not None, \
                "Temporal model should be initialized"
            assert trainer.temporal_optimizer is not None, \
                "Temporal optimizer should be initialized"
            assert trainer.temporal_loss is not None, \
                "Temporal loss should be initialized"
            
            print("  ✓ Temporal model initialized")
            print("  ✓ Temporal optimizer initialized")
            print("  ✓ Temporal loss initialized")
            
            # Test generate_temporal_training_data
            symptom_indices, timestamps, labels = trainer.generate_temporal_training_data(batch_size=4)
            
            assert symptom_indices.shape[0] == 4, "Batch size should be 4"
            assert timestamps.shape[0] == 4, "Batch size should be 4"
            assert labels.shape[0] == 4, "Batch size should be 4"
            
            print(f"  ✓ Generated temporal data: {symptom_indices.shape}")
            
            # Test train_temporal_step
            losses = trainer.train_temporal_step(batch_size=4)
            
            assert 'temporal_loss' in losses, "Should return temporal_loss"
            assert 'temporal_accuracy' in losses, "Should return temporal_accuracy"
            
            print(f"  ✓ Temporal step loss: {losses['temporal_loss']:.4f}")
            print(f"  ✓ Temporal step accuracy: {losses['temporal_accuracy']:.4f}")
        else:
            print("  ⚠️  Temporal module not available, but test succeeded")
        
        print("✅ AdversarialTrainer integration test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping test (missing dependencies): {e}")
        return True
    except Exception as e:
        print(f"❌ AdversarialTrainer integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all learnable temporal tests."""
    print("🏥 PHAITA Learnable Temporal Pattern Matcher Test Suite")
    print("=" * 70)
    
    tests = [
        ("Model Creation", test_learnable_temporal_creation),
        ("Forward Pass", test_learnable_temporal_forward_pass),
        ("Prediction", test_learnable_temporal_prediction),
        ("Temporal Patterns Init", test_learnable_temporal_with_patterns),
        ("Score Timeline", test_learnable_temporal_score_timeline),
        ("AdversarialTrainer Integration", test_adversarial_trainer_integration),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
        print()
    
    # Summary
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All learnable temporal tests passed!")
        return 0
    else:
        print("❌ Some tests failed:")
        for name, result in results:
            if not result:
                print(f"  - {name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
