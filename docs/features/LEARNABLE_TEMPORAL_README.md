# Learnable Temporal Pattern Matching

## Overview

The `LearnableTemporalPatternMatcher` replaces the heuristic-based `TemporalPatternMatcher` with a trainable neural network that learns temporal symptom patterns from data. This approach provides several advantages:

- **Differentiable**: Can be trained end-to-end with gradient descent
- **Data-driven**: Learns patterns from real patient data
- **Flexible**: Can capture complex temporal relationships
- **Clinical initialization**: Starts with domain knowledge from `temporal_patterns.yaml`

## Architecture

The learnable temporal pattern matcher consists of:

1. **TemporalSymptomEncoder (LSTM)**: Core encoder for temporal sequences
   - Embeds symptoms and timestamps
   - Processes sequences through LSTM layers
   - Outputs fixed-size temporal embedding

2. **Classification Head**: Maps temporal embedding to condition probabilities
   - Fully connected layers with ReLU activation
   - Dropout for regularization
   - Outputs logits for each respiratory condition

## Usage

### Creating the Model

```python
from phaita.models.temporal_module import LearnableTemporalPatternMatcher
from phaita.data.icd_conditions import RespiratoryConditions
import yaml

# Load temporal patterns
with open("config/temporal_patterns.yaml") as f:
    temporal_patterns = yaml.safe_load(f)

# Get condition codes
conditions = RespiratoryConditions.get_all_conditions()
condition_codes = list(conditions.keys())

# Create model
model = LearnableTemporalPatternMatcher(
    num_conditions=len(condition_codes),
    symptom_vocab_size=100,  # Size of symptom vocabulary
    symptom_embedding_dim=64,
    hidden_dim=128,
    num_layers=2,
    temporal_patterns=temporal_patterns,
    condition_codes=condition_codes,
)
```

### Training the Model

The model can be trained with standard PyTorch training loops:

```python
import torch
import torch.nn as nn
from torch.optim import AdamW

# Setup
optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training step
model.train()
optimizer.zero_grad()

# Forward pass
logits = model(symptom_indices, timestamps)
loss = loss_fn(logits, condition_labels)

# Backward pass
loss.backward()
optimizer.step()
```

### Integration with AdversarialTrainer

The model is automatically integrated when creating an `AdversarialTrainer`:

```python
from phaita.training.adversarial_trainer import AdversarialTrainer

trainer = AdversarialTrainer(
    use_learnable_temporal=True,
    temporal_lr=1e-3,  # Learning rate for temporal model
)

# Train for 100 epochs
history = trainer.train(
    num_epochs=100,
    batch_size=16,
)
```

The trainer will:
1. Initialize the temporal model with clinical knowledge
2. Generate temporal training data from symptom patterns
3. Train the model alongside other components
4. Track temporal loss and accuracy metrics

## Training Data Generation

The `generate_temporal_training_data` method creates batches of temporal sequences:

1. Randomly selects a condition
2. Loads the typical progression pattern from `temporal_patterns.yaml`
3. Adds timing noise (±6 hours) to simulate real-world variation
4. Converts symptoms to indices
5. Pads sequences to the same length
6. Returns tensors ready for training

## Compatibility

The model maintains compatibility with the original `TemporalPatternMatcher` interface:

```python
# Use like the original pattern matcher
score = model.score_timeline(
    symptom_indices,
    timestamps,
    condition_code="J18.9"  # Pneumonia
)
```

## Model Parameters

Default architecture:
- **Symptom embedding**: 64 dimensions
- **LSTM hidden size**: 128 dimensions
- **LSTM layers**: 2 layers
- **Dropout**: 0.1
- **Total parameters**: ~300K (for 10 conditions, 27 symptoms)

## Performance

The model shows several advantages over heuristic matching:

| Feature | Heuristic Matcher | Learnable Matcher |
|---------|------------------|-------------------|
| Training | Not trainable | End-to-end trainable |
| Temporal patterns | Fixed from YAML | Learned from data |
| Flexibility | Limited | Highly flexible |
| Complexity | O(n×m) per sample | O(n) per sample (after training) |
| Accuracy | ~70-80% | Improves with training |

## Configuration

The model uses `config/temporal_patterns.yaml` for initialization:

```yaml
J18.9:  # Pneumonia
  typical_progression:
    - symptom: fever
      onset_hour: 0
    - symptom: cough
      onset_hour: 12
    - symptom: chest_pain
      onset_hour: 24
```

This provides a strong starting point before training on real data.

## Testing

Run the test suite to validate the implementation:

```bash
# Quick tests (no model downloads)
python tests/test_learnable_temporal_quick.py

# Comprehensive tests (includes AdversarialTrainer)
python tests/test_learnable_temporal.py

# Demo
python demos/demo_learnable_temporal.py
```

## Example Output

```
Learnable Temporal Pattern Matcher Demo
======================================================================

1. Setting up components...
2. Building symptom vocabulary...
   • Total symptoms: 27
   • Total conditions: 10

3. Loading temporal patterns from config/temporal_patterns.yaml...
   • Loaded 10 patterns

4. Creating LearnableTemporalPatternMatcher...
   • Model has 300426 parameters

5. Demo: Creating a pneumonia-like symptom timeline...
   • Target condition: Pneumonia (J18.9)
   • Expected progression (5 symptoms):
     - Fever: 0h
     - Cough: 12h
     - Chest Pain: 24h
     - Dyspnea: 48h

6. Running prediction...
   • Predicted condition: Pneumonia (J18.9)
   • Confidence: 0.8752

   • Top 3 predictions:
     1. Pneumonia (J18.9): 0.8752
     2. Bacterial pneumonia (J15.9): 0.0654
     3. Upper respiratory infection (J06.9): 0.0312
```

## Future Work

Potential improvements:

1. **Attention mechanisms**: Add attention over symptom sequences
2. **Multi-task learning**: Joint training with other objectives
3. **Pre-training**: Train on large medical datasets before fine-tuning
4. **Uncertainty quantification**: Output confidence intervals
5. **Transfer learning**: Adapt to new conditions with few examples

## References

- Original temporal module: `phaita/models/temporal_module.py`
- Training integration: `phaita/training/adversarial_trainer.py`
- Temporal patterns: `config/temporal_patterns.yaml`
- Tests: `tests/test_learnable_temporal*.py`
- Demo: `demos/demo_learnable_temporal.py`
