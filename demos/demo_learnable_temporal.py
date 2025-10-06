#!/usr/bin/env python3
"""
Demo of the learnable temporal pattern matcher.
Shows how to use the new neural network approach for temporal pattern matching.
"""

import torch
import yaml
from pathlib import Path
from phaita.models.temporal_module import LearnableTemporalPatternMatcher
from phaita.data.icd_conditions import RespiratoryConditions
from phaita.models.bayesian_network import BayesianSymptomNetwork

print("=" * 70)
print("Learnable Temporal Pattern Matcher Demo")
print("=" * 70)
print()

# Setup
print("1. Setting up components...")
conditions = RespiratoryConditions.get_all_conditions()
condition_codes = list(conditions.keys())
bayesian_network = BayesianSymptomNetwork()

# Build symptom vocabulary
print("2. Building symptom vocabulary...")
symptom_vocab = set()
for code in condition_codes:
    symptoms = bayesian_network.sample_symptoms(code)
    symptom_vocab.update(symptoms)
symptom_vocab = sorted(list(symptom_vocab))
symptom_to_idx = {symptom: idx + 1 for idx, symptom in enumerate(symptom_vocab)}
symptom_to_idx['<PAD>'] = 0

print(f"   • Total symptoms: {len(symptom_vocab)}")
print(f"   • Total conditions: {len(condition_codes)}")
print()

# Load temporal patterns
print("3. Loading temporal patterns from config/temporal_patterns.yaml...")
config_path = Path(__file__).parent.parent / "config" / "temporal_patterns.yaml"
if config_path.exists():
    with open(config_path) as f:
        temporal_patterns = yaml.safe_load(f)
    print(f"   • Loaded {len(temporal_patterns)} patterns")
else:
    temporal_patterns = {}
    print("   • No patterns found, using default")
print()

# Create model
print("4. Creating LearnableTemporalPatternMatcher...")
model = LearnableTemporalPatternMatcher(
    num_conditions=len(condition_codes),
    symptom_vocab_size=len(symptom_vocab) + 1,
    symptom_embedding_dim=64,
    hidden_dim=128,
    num_layers=2,
    temporal_patterns=temporal_patterns,
    condition_codes=condition_codes,
)
model.eval()
print(f"   • Model has {sum(p.numel() for p in model.parameters())} parameters")
print()

# Demo: Create a pneumonia-like timeline
print("5. Demo: Creating a pneumonia-like symptom timeline...")
condition_code = "J18.9"  # Pneumonia
condition_name = conditions[condition_code]['name']
print(f"   • Target condition: {condition_name} ({condition_code})")

if condition_code in temporal_patterns:
    pattern = temporal_patterns[condition_code]['typical_progression']
    print(f"   • Expected progression ({len(pattern)} symptoms):")
    
    symptom_indices = []
    timestamps = []
    
    for event in pattern:
        symptom = event['symptom']
        onset_hour = event['onset_hour']
        
        if symptom in symptom_to_idx:
            symptom_idx = symptom_to_idx[symptom]
            symptom_indices.append(symptom_idx)
            timestamps.append(float(onset_hour))
            print(f"     - {symptom.replace('_', ' ').title()}: {onset_hour}h")
    
    # Convert to tensors (batch size 1)
    symptom_indices_tensor = torch.tensor([symptom_indices], dtype=torch.long)
    timestamps_tensor = torch.tensor([timestamps], dtype=torch.float)
    
    print()
    print("6. Running prediction...")
    
    # Predict
    with torch.no_grad():
        predicted_indices, probs = model.predict_condition(
            symptom_indices_tensor, timestamps_tensor
        )
        predicted_idx = predicted_indices[0].item()
        predicted_code = condition_codes[predicted_idx]
        predicted_name = conditions[predicted_code]['name']
        predicted_prob = probs[0, predicted_idx].item()
    
    print(f"   • Predicted condition: {predicted_name} ({predicted_code})")
    print(f"   • Confidence: {predicted_prob:.4f}")
    print()
    
    # Show top 3 predictions
    print("   • Top 3 predictions:")
    top_probs, top_indices = torch.topk(probs[0], k=min(3, len(condition_codes)))
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        code = condition_codes[idx]
        name = conditions[code]['name']
        print(f"     {i+1}. {name} ({code}): {prob.item():.4f}")
    
    print()
    print("7. Testing score_timeline compatibility method...")
    score = model.score_timeline(symptom_indices_tensor, timestamps_tensor, condition_code)
    print(f"   • Score for {condition_name}: {score:.4f}")

print()
print("=" * 70)
print("Demo complete!")
print()
print("Key Features:")
print("  ✓ Learnable neural network replaces heuristics")
print("  ✓ Uses LSTM encoder for temporal patterns")
print("  ✓ Can be trained with standard cross-entropy loss")
print("  ✓ Initialized with clinical knowledge from YAML")
print("  ✓ Compatible with existing TemporalPatternMatcher interface")
print("=" * 70)
