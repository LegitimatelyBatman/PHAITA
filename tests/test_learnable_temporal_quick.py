#!/usr/bin/env python3
"""
Quick test for LearnableTemporalPatternMatcher without model downloads.
Tests just the core functionality without initializing full AdversarialTrainer.
"""

import sys
import traceback


def test_temporal_data_generation():
    """Test temporal data generation method standalone."""
    print("🧪 Testing temporal data generation...")
    
    try:
        import torch
        import yaml
        from pathlib import Path
        import random
        from phaita.data.icd_conditions import RespiratoryConditions
        from phaita.models.bayesian_network import BayesianSymptomNetwork
        from phaita.models.temporal_module import LearnableTemporalPatternMatcher
        
        # Setup basic components
        conditions = RespiratoryConditions.get_all_conditions()
        condition_codes = list(conditions.keys())
        bayesian_network = BayesianSymptomNetwork()
        
        # Build symptom vocabulary
        symptom_vocab = set()
        for code in condition_codes:
            symptoms = bayesian_network.sample_symptoms(code)
            symptom_vocab.update(symptoms)
        symptom_vocab = sorted(list(symptom_vocab))
        symptom_to_idx = {symptom: idx + 1 for idx, symptom in enumerate(symptom_vocab)}
        symptom_to_idx['<PAD>'] = 0
        
        # Create temporal model
        model = LearnableTemporalPatternMatcher(
            num_conditions=len(condition_codes),
            symptom_vocab_size=len(symptom_vocab) + 1,
            condition_codes=condition_codes,
        )
        
        # Load temporal patterns
        config_path = Path(__file__).parent.parent / "config" / "temporal_patterns.yaml"
        if config_path.exists():
            with open(config_path) as f:
                temporal_patterns = yaml.safe_load(f)
        else:
            temporal_patterns = {}
        
        # Simulate generate_temporal_training_data
        batch_size = 4
        all_symptom_indices = []
        all_timestamps = []
        all_condition_labels = []
        
        for _ in range(batch_size):
            condition_code = random.choice(condition_codes)
            condition_idx = model.condition_to_idx[condition_code]
            
            if condition_code in temporal_patterns:
                pattern = temporal_patterns[condition_code].get('typical_progression', [])
            else:
                symptoms = bayesian_network.sample_symptoms(condition_code)
                pattern = [
                    {'symptom': symptom, 'onset_hour': i * 12}
                    for i, symptom in enumerate(symptoms)
                ]
            
            symptom_indices = []
            timestamps = []
            
            for event in pattern:
                symptom = event['symptom']
                base_time = event['onset_hour']
                noisy_time = base_time + random.uniform(-6, 6)
                noisy_time = max(0, noisy_time)
                
                if symptom in symptom_to_idx:
                    symptom_idx = symptom_to_idx[symptom]
                    symptom_indices.append(symptom_idx)
                    timestamps.append(noisy_time)
            
            if not symptom_indices:
                symptom_indices = [1]
                timestamps = [0.0]
            
            all_symptom_indices.append(symptom_indices)
            all_timestamps.append(timestamps)
            all_condition_labels.append(condition_idx)
        
        # Pad sequences
        max_len = max(len(seq) for seq in all_symptom_indices)
        padded_symptom_indices = []
        padded_timestamps = []
        
        for symptom_seq, time_seq in zip(all_symptom_indices, all_timestamps):
            padded_symptoms = symptom_seq + [0] * (max_len - len(symptom_seq))
            padded_times = time_seq + [0.0] * (max_len - len(time_seq))
            padded_symptom_indices.append(padded_symptoms)
            padded_timestamps.append(padded_times)
        
        # Convert to tensors
        symptom_indices_tensor = torch.tensor(padded_symptom_indices, dtype=torch.long)
        timestamps_tensor = torch.tensor(padded_timestamps, dtype=torch.float)
        condition_labels_tensor = torch.tensor(all_condition_labels, dtype=torch.long)
        
        # Validate shapes
        assert symptom_indices_tensor.shape[0] == batch_size
        assert timestamps_tensor.shape[0] == batch_size
        assert condition_labels_tensor.shape[0] == batch_size
        assert symptom_indices_tensor.shape == timestamps_tensor.shape
        
        print(f"  ✓ Generated {batch_size} temporal sequences")
        print(f"  ✓ Symptom indices shape: {symptom_indices_tensor.shape}")
        print(f"  ✓ Timestamps shape: {timestamps_tensor.shape}")
        print(f"  ✓ Labels shape: {condition_labels_tensor.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            logits = model(symptom_indices_tensor, timestamps_tensor)
            predictions = torch.argmax(logits, dim=-1)
        
        assert logits.shape == (batch_size, len(condition_codes))
        assert predictions.shape == (batch_size,)
        
        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Logits shape: {logits.shape}")
        print(f"  ✓ Predictions: {predictions.tolist()}")
        
        print("✅ Temporal data generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Temporal data generation test failed: {e}")
        traceback.print_exc()
        return False


def test_temporal_training_step():
    """Test temporal training step standalone."""
    print("🧪 Testing temporal training step...")
    
    try:
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        import yaml
        from pathlib import Path
        import random
        from phaita.data.icd_conditions import RespiratoryConditions
        from phaita.models.bayesian_network import BayesianSymptomNetwork
        from phaita.models.temporal_module import LearnableTemporalPatternMatcher
        
        # Setup basic components
        conditions = RespiratoryConditions.get_all_conditions()
        condition_codes = list(conditions.keys())
        bayesian_network = BayesianSymptomNetwork()
        
        # Build symptom vocabulary
        symptom_vocab = set()
        for code in condition_codes:
            symptoms = bayesian_network.sample_symptoms(code)
            symptom_vocab.update(symptoms)
        symptom_vocab = sorted(list(symptom_vocab))
        symptom_to_idx = {symptom: idx + 1 for idx, symptom in enumerate(symptom_vocab)}
        symptom_to_idx['<PAD>'] = 0
        
        # Create temporal model
        model = LearnableTemporalPatternMatcher(
            num_conditions=len(condition_codes),
            symptom_vocab_size=len(symptom_vocab) + 1,
            condition_codes=condition_codes,
        )
        
        # Create optimizer and loss
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        # Load temporal patterns
        config_path = Path(__file__).parent.parent / "config" / "temporal_patterns.yaml"
        if config_path.exists():
            with open(config_path) as f:
                temporal_patterns = yaml.safe_load(f)
        else:
            temporal_patterns = {}
        
        # Generate training data (simplified version)
        batch_size = 8
        all_symptom_indices = []
        all_timestamps = []
        all_condition_labels = []
        
        for _ in range(batch_size):
            condition_code = random.choice(condition_codes)
            condition_idx = model.condition_to_idx[condition_code]
            
            if condition_code in temporal_patterns:
                pattern = temporal_patterns[condition_code].get('typical_progression', [])[:3]  # Take first 3
            else:
                symptoms = bayesian_network.sample_symptoms(condition_code)[:3]
                pattern = [
                    {'symptom': symptom, 'onset_hour': i * 12}
                    for i, symptom in enumerate(symptoms)
                ]
            
            symptom_indices = []
            timestamps = []
            
            for event in pattern:
                symptom = event['symptom']
                base_time = event['onset_hour']
                
                if symptom in symptom_to_idx:
                    symptom_idx = symptom_to_idx[symptom]
                    symptom_indices.append(symptom_idx)
                    timestamps.append(float(base_time))
            
            if not symptom_indices:
                symptom_indices = [1]
                timestamps = [0.0]
            
            all_symptom_indices.append(symptom_indices)
            all_timestamps.append(timestamps)
            all_condition_labels.append(condition_idx)
        
        # Pad sequences
        max_len = max(len(seq) for seq in all_symptom_indices)
        padded_symptom_indices = []
        padded_timestamps = []
        
        for symptom_seq, time_seq in zip(all_symptom_indices, all_timestamps):
            padded_symptoms = symptom_seq + [0] * (max_len - len(symptom_seq))
            padded_times = time_seq + [0.0] * (max_len - len(time_seq))
            padded_symptom_indices.append(padded_symptoms)
            padded_timestamps.append(padded_times)
        
        # Convert to tensors
        symptom_indices_tensor = torch.tensor(padded_symptom_indices, dtype=torch.long)
        timestamps_tensor = torch.tensor(padded_timestamps, dtype=torch.float)
        condition_labels_tensor = torch.tensor(all_condition_labels, dtype=torch.long)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        logits = model(symptom_indices_tensor, timestamps_tensor)
        loss = loss_fn(logits, condition_labels_tensor)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == condition_labels_tensor).float().mean().item()
        
        print(f"  ✓ Training step completed")
        print(f"  ✓ Loss: {loss.item():.4f}")
        print(f"  ✓ Accuracy: {accuracy:.4f}")
        print(f"  ✓ Gradients computed and applied")
        
        print("✅ Temporal training step test passed")
        return True
        
    except Exception as e:
        print(f"❌ Temporal training step test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run quick temporal tests."""
    print("🏥 PHAITA Quick Learnable Temporal Tests")
    print("=" * 70)
    
    tests = [
        ("Temporal Data Generation", test_temporal_data_generation),
        ("Temporal Training Step", test_temporal_training_step),
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
        print("✅ All quick temporal tests passed!")
        return 0
    else:
        print("❌ Some tests failed:")
        for name, result in results:
            if not result:
                print(f"  - {name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
