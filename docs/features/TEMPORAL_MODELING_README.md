# Temporal Symptom Modeling

This module implements temporal symptom tracking and pattern matching for improved diagnosis accuracy in PHAITA.

## Overview

The temporal modeling feature tracks when symptoms appear over time and matches patient symptom progressions against known disease patterns. This helps identify conditions that follow characteristic temporal progressions (e.g., pneumonia typically starts with fever, followed by cough 12 hours later, then chest pain and dyspnea).

## Components

### 1. SymptomTimeline

Tracks symptom onset times and maintains chronological order.

```python
from phaita.models.temporal_module import SymptomTimeline

timeline = SymptomTimeline()
timeline.add_symptom("fever", 0)      # 0 hours ago (initial symptom)
timeline.add_symptom("cough", 12)     # 12 hours after first symptom
timeline.add_symptom("chest_pain", 24) # 24 hours after first symptom

# Get chronological progression
progression = timeline.get_progression_pattern()
# Returns: [("fever", 0), ("cough", 12), ("chest_pain", 24)]
```

**Methods:**
- `add_symptom(symptom, hours_ago)` - Add a symptom with its onset time
- `get_progression_pattern()` - Get chronologically ordered (symptom, hours) tuples
- `get_symptom_order()` - Get just symptom names in order
- `clear()` - Clear all events

### 2. TemporalPatternMatcher

Scores patient timelines against canonical disease progressions.

```python
from phaita.models.temporal_module import TemporalPatternMatcher
import yaml

# Load patterns from config
with open("config/temporal_patterns.yaml") as f:
    patterns = yaml.safe_load(f)

matcher = TemporalPatternMatcher(patterns)

# Score a patient timeline against pneumonia pattern
score = matcher.score_timeline(timeline, "J18.9")  # J18.9 = Pneumonia
# Returns: float in range [0.5, 1.5]
#   > 1.0 = Good match (symptoms in expected order)
#   = 1.0 = Neutral (no pattern or insufficient data)
#   < 1.0 = Poor match (wrong order/timing)
```

**Scoring Algorithm:**
1. **Temporal Alignment**: Exponential decay based on time difference from expected onset
   - Perfect match (0 hours difference) = 1.0
   - 24 hours difference ≈ 0.6
   - 48 hours difference ≈ 0.4
2. **Order Bonus**: Additional score for symptoms appearing in correct chronological sequence
3. **Final Score**: Base alignment + order bonus, scaled to [0.5, 1.5]

### 3. TemporalSymptomEncoder (LSTM)

Deep learning encoder for symptom sequences with temporal information.

```python
from phaita.models.temporal_module import TemporalSymptomEncoder
import torch

# Create encoder (requires PyTorch)
encoder = TemporalSymptomEncoder(
    symptom_vocab_size=100,
    symptom_embedding_dim=64,
    hidden_dim=128,
    num_layers=2
)

# Encode symptom sequences
symptom_indices = torch.randint(0, 100, (batch_size, seq_len))
timestamps = torch.rand(batch_size, seq_len) * 100  # hours

embeddings = encoder(symptom_indices, timestamps)
# Returns: [batch_size, hidden_dim] temporal embeddings
```

**Architecture:**
- Symptom embedding layer (vocab_size → embedding_dim)
- Time projection (1D timestamp → embedding_dim)
- LSTM layers (combines symptom + time → hidden state)
- Output projection (hidden → output_dim)

### 4. DialogueEngine Integration

The `DialogueEngine` now supports temporal information in belief updates.

```python
from phaita.conversation.dialogue_engine import DialogueEngine

# Initialize with temporal module enabled
engine = DialogueEngine(use_temporal_module=True)

# Update beliefs with temporal information
engine.update_beliefs("fever", True, hours_since_onset=0)
engine.update_beliefs("cough", True, hours_since_onset=12)
engine.update_beliefs("chest_pain", True, hours_since_onset=24)

# Temporal pattern matching automatically adjusts probabilities
differential = engine.get_differential_diagnosis()
# Conditions with matching temporal patterns get boosted probabilities
```

**Parameters:**
- `use_temporal_module` (bool): Enable temporal pattern matching (default: True)

**New Method Signature:**
```python
def update_beliefs(
    self,
    symptom: str,
    present: bool,
    hours_since_onset: Optional[float] = None
) -> None:
    """Update condition probabilities using Bayes' rule with temporal info."""
```

## Configuration

### temporal_patterns.yaml

Defines typical symptom progressions for each condition:

```yaml
J18.9:  # Pneumonia
  typical_progression:
    - symptom: fever
      onset_hour: 0          # Initial symptom
    - symptom: cough
      onset_hour: 12         # 12 hours after fever
    - symptom: chest_pain
      onset_hour: 24         # 24 hours after fever
    - symptom: dyspnea
      onset_hour: 48         # 48 hours after fever
```

**Included Conditions:**
- J18.9: Pneumonia
- J45.9: Asthma
- J44.9: COPD
- J20.9: Acute Bronchitis
- J06.9: Upper Respiratory Infection
- J93.0: Pneumothorax
- J81.0: Pulmonary Edema
- J15.9: Bacterial Pneumonia
- J12.9: Viral Pneumonia
- J21.9: Bronchiolitis

## Testing

Run the comprehensive test suite:

```bash
python test_temporal_modeling.py
```

**Test Coverage:**
- ✅ SymptomTimeline tracking and ordering
- ✅ TemporalPatternMatcher scoring
- ✅ LSTM encoder forward pass
- ✅ DialogueEngine integration
- ✅ Accuracy improvement validation

## Demo

Run the interactive demo:

```bash
python demo_temporal_modeling.py
```

**Demo Output:**
```
📋 Demo 1: Symptom Timeline Tracking
Patient presents with symptoms over time:
  • Fever: 3 days ago (0h ago)
  • Cough: 2 days ago (36h ago)
  • Chest Pain: 1.5 days ago (48h ago)
  • Dyspnea: This morning (72h ago)

🔍 Demo 2: Temporal Pattern Matching
Testing patient timeline against known conditions:
  Pneumonia                 Score: 1.312  🟢 Excellent match
  Asthma                    Score: 0.910  🟠 Weak match
  COPD                      Score: 1.151  🟡 Good match
  Acute Bronchitis          Score: 0.913  🟠 Weak match

🎯 Best temporal match: Pneumonia (score: 1.312)
```

## Requirements

**Core Features (No Dependencies):**
- SymptomTimeline
- TemporalPatternMatcher
- YAML pattern loading

**Deep Learning Features (Requires PyTorch):**
- TemporalSymptomEncoder (LSTM)

**Installation:**
```bash
# Core features only
pip install pyyaml

# Full features with deep learning
pip install torch pyyaml
```

## Performance

- **Timeline Operations**: O(n log n) for add_symptom (sorting)
- **Pattern Matching**: O(m * n) where m = symptoms in pattern, n = symptoms in timeline
- **LSTM Encoding**: ~1ms per batch on CPU, <0.1ms on GPU

## Extending

### Adding New Conditions

1. Open `config/temporal_patterns.yaml`
2. Add new condition with ICD-10 code:

```yaml
J00.0:  # New Condition
  typical_progression:
    - symptom: initial_symptom
      onset_hour: 0
    - symptom: secondary_symptom
      onset_hour: 24
```

### Custom Scoring Algorithms

Subclass `TemporalPatternMatcher` and override `score_timeline()`:

```python
class CustomMatcher(TemporalPatternMatcher):
    def score_timeline(self, timeline, condition_code):
        # Custom scoring logic
        return custom_score
```

## Limitations

1. **Requires Accurate Timing**: Patient must recall symptom onset times
2. **Fixed Patterns**: Assumes consistent disease progressions
3. **Individual Variation**: May not match atypical presentations
4. **Memory Bias**: Patient recall may be imperfect

## Future Enhancements

- [ ] Learn patterns from real patient data
- [ ] Support for symptom duration (not just onset)
- [ ] Probabilistic pattern matching (distributions)
- [ ] Multi-modal temporal features (lab results, vital signs)
- [ ] Transformer-based encoding (attention over time)

## References

- Problem statement: See issue #[number]
- Implementation details: `IMPLEMENTATION_DETAILS.md`
- API documentation: Code docstrings

## License

Part of the PHAITA project. See main repository LICENSE.
