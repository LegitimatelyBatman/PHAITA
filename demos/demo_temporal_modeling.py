#!/usr/bin/env python3
"""
Demo script for temporal symptom modeling.

This demonstrates how the temporal module tracks symptom progression
and uses pattern matching to improve diagnosis accuracy.
"""

import sys
from pathlib import Path
import importlib.util

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import temporal module directly to avoid __init__ dependency issues
def import_temporal_module():
    """Import temporal module directly from file."""
    spec = importlib.util.spec_from_file_location(
        'temporal_module',
        Path(__file__).parent / 'phaita' / 'models' / 'temporal_module.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

temporal_module = import_temporal_module()
SymptomTimeline = temporal_module.SymptomTimeline
TemporalPatternMatcher = temporal_module.TemporalPatternMatcher
TemporalSymptomEncoder = temporal_module.TemporalSymptomEncoder


def _get_score_indicator(score):
    """Return visual indicator for score quality."""
    if score > 1.2:
        return "🟢 Excellent match"
    elif score > 1.0:
        return "🟡 Good match"
    elif score == 1.0:
        return "⚪ Neutral"
    elif score > 0.8:
        return "🟠 Weak match"
    else:
        return "🔴 Poor match"


print("=" * 70)
print("PHAITA Temporal Symptom Modeling Demo")
print("=" * 70)
print()

# Demo 1: SymptomTimeline
print("📋 Demo 1: Symptom Timeline Tracking")
print("-" * 70)

timeline = SymptomTimeline()
print("Patient presents with symptoms over time:")
print()

# Add symptoms as they appeared
symptoms_over_time = [
    ("fever", 0, "3 days ago"),
    ("cough", 36, "2 days ago"),
    ("chest_pain", 48, "1.5 days ago"),
    ("dyspnea", 72, "This morning"),
]

for symptom, hours, description in symptoms_over_time:
    timeline.add_symptom(symptom, hours)
    print(f"  • {symptom.replace('_', ' ').title()}: {description} ({hours}h ago)")

print()
print("Chronological progression (earliest → latest):")
progression = timeline.get_progression_pattern()
for symptom, hours in progression:
    print(f"  {hours}h: {symptom.replace('_', ' ')}")

print()

# Demo 2: Pattern Matching
print("🔍 Demo 2: Temporal Pattern Matching")
print("-" * 70)
import yaml

# Load temporal patterns
patterns_path = Path(__file__).parent / "config" / "temporal_patterns.yaml"
with open(patterns_path) as f:
    patterns = yaml.safe_load(f)

matcher = TemporalPatternMatcher(patterns)

print("Testing patient timeline against known conditions:")
print()

# Test against different conditions
conditions_to_test = [
    ("J18.9", "Pneumonia"),
    ("J45.9", "Asthma"),
    ("J44.9", "COPD"),
    ("J20.9", "Acute Bronchitis"),
]

scores = []
for code, name in conditions_to_test:
    score = matcher.score_timeline(timeline, code)
    scores.append((name, score, code))
    print(f"  {name:25} Score: {score:.3f}  {_get_score_indicator(score)}")

print()
print("Interpretation:")
print("  • Score > 1.0: Good temporal match (symptoms appeared in expected order)")
print("  • Score = 1.0: Neutral (no pattern defined or insufficient data)")
print("  • Score < 1.0: Poor match (symptoms appeared in wrong order/timing)")
print()

# Show best match
best_match = max(scores, key=lambda x: x[1])
print(f"🎯 Best temporal match: {best_match[0]} (score: {best_match[1]:.3f})")
print()

# Demo 3: Expected progression for best match
print(f"📊 Demo 3: Expected Progression for {best_match[0]}")
print("-" * 70)

expected = patterns[best_match[2]]["typical_progression"]
print("Typical symptom progression:")
for event in expected:
    symptom = event["symptom"].replace("_", " ").title()
    onset = event["onset_hour"]
    if onset == 0:
        print(f"  Hour 0: {symptom} (initial symptom)")
    else:
        print(f"  +{onset}h: {symptom}")

print()

# Demo 4: LSTM Encoder (if torch available)
print("🧠 Demo 4: LSTM Temporal Encoder")
print("-" * 70)

try:
    import torch
    
    # Create encoder
    encoder = TemporalSymptomEncoder(
        symptom_vocab_size=100,
        symptom_embedding_dim=32,
        hidden_dim=64,
        num_layers=2,
    )
    
    # Create sample data (batch of 2 patients)
    batch_size = 2
    seq_len = 4
    
    # Random symptom indices and timestamps
    symptom_indices = torch.randint(0, 100, (batch_size, seq_len))
    timestamps = torch.tensor([
        [0.0, 12.0, 24.0, 48.0],  # Patient 1
        [0.0, 6.0, 12.0, 24.0],   # Patient 2
    ])
    
    # Encode sequences
    embeddings = encoder(symptom_indices, timestamps)
    
    print("LSTM encoder successfully created:")
    print(f"  • Input: {batch_size} patients × {seq_len} symptoms")
    print(f"  • Output embedding shape: {embeddings.shape}")
    print(f"  • Each patient represented as {embeddings.shape[1]}-dimensional vector")
    print()
    print("✅ LSTM temporal encoder working!")
    
except ImportError as e:
    print("⚠️  PyTorch not available - LSTM encoder demo skipped")
    print(f"   (Error: {e})")
    print("   Install with: pip install torch")

print()
print("=" * 70)
print("✅ Temporal Symptom Modeling Demo Complete!")
print("=" * 70)
