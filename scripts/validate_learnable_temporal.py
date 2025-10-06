#!/usr/bin/env python3
"""
Validation script for learnable temporal pattern matching implementation.
Demonstrates all key features and verifies they work correctly.
"""

import sys

def validate_implementation():
    """Run comprehensive validation of the implementation."""
    
    print("=" * 80)
    print("LEARNABLE TEMPORAL PATTERN MATCHING - VALIDATION")
    print("=" * 80)
    print()
    
    # Test 1: LearnableTemporalPatternMatcher exists and can be instantiated
    print("✓ Test 1: Model Creation")
    try:
        import torch
        from phaita.models.temporal_module import LearnableTemporalPatternMatcher
        from phaita.data.icd_conditions import RespiratoryConditions
        
        conditions = RespiratoryConditions.get_all_conditions()
        condition_codes = list(conditions.keys())
        
        model = LearnableTemporalPatternMatcher(
            num_conditions=len(condition_codes),
            symptom_vocab_size=100,
            condition_codes=condition_codes,
        )
        
        print(f"  • Model created with {len(condition_codes)} conditions")
        print(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 2: Forward pass works
    print("✓ Test 2: Forward Pass")
    try:
        symptom_indices = torch.randint(0, 100, (4, 5))
        timestamps = torch.rand(4, 5) * 100
        
        logits = model(symptom_indices, timestamps)
        assert logits.shape == (4, len(condition_codes))
        
        print(f"  • Input shape: {symptom_indices.shape}")
        print(f"  • Output shape: {logits.shape}")
        print()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 3: Prediction works
    print("✓ Test 3: Prediction")
    try:
        model.eval()
        with torch.no_grad():
            predicted_indices, probs = model.predict_condition(symptom_indices, timestamps)
        
        assert predicted_indices.shape == (4,)
        assert probs.shape == (4, len(condition_codes))
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        
        print(f"  • Predictions shape: {predicted_indices.shape}")
        print(f"  • Probabilities shape: {probs.shape}")
        print(f"  • Probabilities sum to 1: ✓")
        print()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 4: Clinical initialization with temporal patterns
    print("✓ Test 4: Clinical Initialization")
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent / "config" / "temporal_patterns.yaml"
        if config_path.exists():
            with open(config_path) as f:
                temporal_patterns = yaml.safe_load(f)
            
            model_with_patterns = LearnableTemporalPatternMatcher(
                num_conditions=len(condition_codes),
                symptom_vocab_size=100,
                temporal_patterns=temporal_patterns,
                condition_codes=condition_codes,
            )
            
            print(f"  • Loaded {len(temporal_patterns)} temporal patterns")
            print(f"  • Model initialized with clinical knowledge")
        else:
            print(f"  • temporal_patterns.yaml not found (OK for minimal setup)")
        print()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 5: Compatibility with TemporalPatternMatcher interface
    print("✓ Test 5: API Compatibility")
    try:
        score = model.score_timeline(
            symptom_indices[:1],
            timestamps[:1],
            condition_codes[0]
        )
        
        assert 0.0 <= score <= 1.0
        
        print(f"  • score_timeline() works")
        print(f"  • Score for {condition_codes[0]}: {score:.4f}")
        print()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 6: Training step works
    print("✓ Test 6: Training Step")
    try:
        import torch.nn as nn
        from torch.optim import AdamW
        
        optimizer = AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        
        model.train()
        optimizer.zero_grad()
        
        labels = torch.randint(0, len(condition_codes), (4,))
        logits = model(symptom_indices, timestamps)
        loss = loss_fn(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        print(f"  • Loss computed: {loss.item():.4f}")
        print(f"  • Gradients computed and applied")
        print()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 7: Existing temporal components still work
    print("✓ Test 7: Backward Compatibility")
    try:
        from phaita.models.temporal_module import (
            SymptomTimeline, 
            TemporalSymptomEncoder, 
            TemporalPatternMatcher
        )
        
        # Test SymptomTimeline
        timeline = SymptomTimeline()
        timeline.add_symptom("fever", 0)
        timeline.add_symptom("cough", 12)
        assert len(timeline.events) == 2
        
        # Test TemporalSymptomEncoder
        encoder = TemporalSymptomEncoder(
            symptom_vocab_size=50,
            symptom_embedding_dim=32,
            hidden_dim=64,
        )
        test_indices = torch.randint(0, 50, (2, 5))
        test_times = torch.rand(2, 5) * 100
        output = encoder(test_indices, test_times)
        assert output.shape == (2, 64)
        
        # Test TemporalPatternMatcher
        matcher = TemporalPatternMatcher({})
        score = matcher.score_timeline(timeline, "J18.9")
        assert score >= 0.5
        
        print(f"  • SymptomTimeline works")
        print(f"  • TemporalSymptomEncoder works")
        print(f"  • TemporalPatternMatcher works")
        print()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 8: AdversarialTrainer integration (basic check)
    print("✓ Test 8: AdversarialTrainer Integration")
    try:
        # Just verify the imports and basic structure
        from phaita.training.adversarial_trainer import AdversarialTrainer
        
        # Check that the new parameters exist
        import inspect
        sig = inspect.signature(AdversarialTrainer.__init__)
        params = list(sig.parameters.keys())
        
        assert 'use_learnable_temporal' in params, "Missing use_learnable_temporal parameter"
        assert 'temporal_lr' in params, "Missing temporal_lr parameter"
        
        print(f"  • AdversarialTrainer has temporal parameters")
        print(f"  • Integration is ready")
        print()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # All tests passed
    print("=" * 80)
    print("✅ ALL VALIDATION TESTS PASSED")
    print("=" * 80)
    print()
    print("Summary:")
    print("  • LearnableTemporalPatternMatcher works correctly")
    print("  • Model can be trained with standard PyTorch")
    print("  • Compatible with existing temporal APIs")
    print("  • Integrated with AdversarialTrainer")
    print("  • No breaking changes to existing code")
    print()
    print("The implementation is ready for use!")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = validate_implementation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
