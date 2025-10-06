# Learnable Comorbidity and Symptom Causality Guide

## Overview

PHAITA now supports **learnable comorbidity effects** and **learnable symptom causality** through PyTorch parameters instead of fixed weights from YAML configuration files. This enables the model to learn optimal weights from data through gradient descent during adversarial training.

## Motivation

Previously, comorbidity effects (e.g., how diabetes affects fatigue) and symptom causality (e.g., fever causes fatigue) were hardcoded in YAML files based on clinical literature. While evidence-based, these fixed values may not generalize optimally across different patient populations or clinical contexts.

**Learnable weights** allow the model to:
- Fine-tune initial clinical values based on observed data
- Adapt to specific patient populations
- Discover relationships not explicitly encoded in medical literature
- Enable end-to-end training of the entire medical triage pipeline

## Architecture

### Learnable Comorbidity Effects

**Module**: `phaita.models.learnable_comorbidity.LearnableComorbidityEffects`

Converts fixed comorbidity symptom modifiers into learnable PyTorch parameters:

```python
from phaita.models.learnable_comorbidity import LearnableComorbidityEffects

# Create learnable comorbidity module
learnable_comorbidity = LearnableComorbidityEffects()

# It's a PyTorch nn.Module with parameters
print(learnable_comorbidity.comorbidity_weights.shape)  # [num_comorbidities, num_symptoms]
print(f"Total parameters: {learnable_comorbidity.comorbidity_weights.numel()}")

# Get learned modifiers for a comorbidity
diabetes_modifiers = learnable_comorbidity.get_symptom_modifiers("diabetes")
print(diabetes_modifiers)  # {'fatigue': 1.32, 'infection_risk': 1.48, ...}
```

**Implementation Details:**
- **Initialization**: Weights initialized from `config/comorbidity_effects.yaml`
- **Parameterization**: Log-multipliers (so `exp(param)` gives positive multipliers)
- **Parameters**: One weight per (comorbidity, symptom) pair (typically ~184 params for 8 comorbidities × 23 symptoms)
- **Gradients**: Fully differentiable for backpropagation

### Learnable Symptom Causality

**Module**: `phaita.models.learnable_causality.LearnableSymptomCausality`

Converts fixed causal and temporal edge strengths into learnable PyTorch parameters:

```python
from phaita.models.learnable_causality import LearnableSymptomCausality

# Create learnable causality module
learnable_causality = LearnableSymptomCausality()

# It's a PyTorch nn.Module with parameters
print(f"Causal edge parameters: {len(learnable_causality.causal_weights)}")
print(f"Temporal edge parameters: {len(learnable_causality.temporal_weights)}")

# Get learned causal edges
causal_edges = learnable_causality.get_causal_edges()
for source, target, strength in causal_edges[:3]:
    print(f"{source} → {target}: {strength:.3f}")

# Get config compatible with GNN
gnn_config = learnable_causality.get_config_for_gnn()
```

**Implementation Details:**
- **Initialization**: Strengths initialized from `config/symptom_causality.yaml`
- **Parameterization**: Logits (so `sigmoid(param)` gives strengths in [0, 1])
- **Parameters**: One weight per causal edge (~10 params) + one weight per temporal edge (~6 params)
- **Temporal delays**: Fixed (not learned) but stored as buffers
- **Gradients**: Fully differentiable for backpropagation

## Integration

### Enhanced Bayesian Network

Use learnable comorbidity in symptom generation:

```python
from phaita.models.enhanced_bayesian_network import create_enhanced_bayesian_network

# Create with learnable comorbidity
network = create_enhanced_bayesian_network(
    use_learnable_comorbidity=True,
    device="cuda"
)

# Sample symptoms (uses learned modifiers)
symptoms, metadata = network.sample_symptoms(
    "J45.9",  # Asthma
    comorbidities=["diabetes", "obesity"],
    age_group="adult",
    severity="moderate"
)

# Access the learnable module
learnable_comorbidity = network.learnable_comorbidity
```

**Backward Compatibility**: Setting `use_learnable_comorbidity=False` (default) uses fixed YAML weights.

### GNN Module

Use learnable causality in symptom graphs:

```python
from phaita.models.gnn_module import SymptomGraphModule
from phaita.models.learnable_causality import LearnableSymptomCausality
from phaita.data.icd_conditions import RespiratoryConditions

conditions = RespiratoryConditions.get_all_conditions()
learnable_causality = LearnableSymptomCausality()

# Create GNN with learnable causality
gnn = SymptomGraphModule(
    conditions=conditions,
    use_causal_edges=True,
    learnable_causality=learnable_causality
)

# Forward pass (uses learned edge strengths)
output = gnn(batch_size=4)
```

**Backward Compatibility**: Omitting `learnable_causality` uses fixed YAML weights.

### Discriminator

The discriminator automatically uses learnable causality if provided:

```python
from phaita.models.discriminator import DiagnosisDiscriminator
from phaita.models.learnable_causality import LearnableSymptomCausality

learnable_causality = LearnableSymptomCausality()

discriminator = DiagnosisDiscriminator(
    use_pretrained=True,
    use_causal_edges=True,
    learnable_causality=learnable_causality
)
```

### Adversarial Trainer

Enable learnable modules in training:

```python
from phaita.training.adversarial_trainer import AdversarialTrainer

trainer = AdversarialTrainer(
    use_pretrained_generator=True,
    use_pretrained_discriminator=True,
    use_learnable_bayesian=True,        # Learn Bayesian priors
    use_learnable_comorbidity=True,      # Learn comorbidity effects
    use_learnable_causality=True,        # Learn causal edge strengths
    bayesian_lr=1e-3,
    comorbidity_lr=1e-3,
    causality_lr=1e-3,
    device="cuda"
)

# Train (will update all learnable weights)
trainer.train(num_epochs=10, batch_size=8)
```

The trainer automatically:
- Creates optimizers for each learnable module
- Computes gradients through the full pipeline
- Updates weights via backpropagation
- Tracks losses in training history

## Training

### Loss Functions

**Comorbidity Effects**: Trained indirectly through:
- Generator loss (symptom realism)
- Discriminator loss (diagnostic accuracy)
- Medical accuracy loss (symptom-condition alignment)

**Symptom Causality**: Trained indirectly through:
- Discriminator loss (diagnostic accuracy via GNN)
- Graph structure influences classification
- Edge weights adapt to improve diagnosis

### Learning Rates

Recommended learning rates (empirically determined):
- `bayesian_lr=1e-3`: Bayesian network parameters
- `comorbidity_lr=1e-3`: Comorbidity effect weights
- `causality_lr=1e-3`: Causal edge strengths

Lower than generator/discriminator LRs to ensure stable convergence.

### Gradient Flow

```
Complaint Text
    ↓
[Discriminator]
    ↓ (text features)
[GNN with Learnable Causality]  ← gradients flow here
    ↓ (graph features)
[Fusion Layer]
    ↓
[Diagnosis Head]
    ↓
Loss (CrossEntropy)
    ↓ (backpropagation)
Updates Causality Weights
```

## Parameters

### Total Learnable Parameters

- **Bayesian Network**: ~20 parameters (symptom probabilities per condition)
- **Comorbidity Effects**: ~184 parameters (8 comorbidities × 23 symptoms)
- **Symptom Causality**: ~16 parameters (10 causal + 6 temporal edges)
- **Total New Parameters**: ~220 (minimal overhead)

Compare to:
- DeBERTa encoder: ~86M parameters
- Generator (Mistral 7B): ~7B parameters

The learnable medical knowledge is a tiny fraction of model size but captures critical domain expertise.

## Testing

Run comprehensive tests:

```bash
# Test learnable modules
python tests/test_learnable_modules.py

# Test enhanced Bayesian (with learnable)
python tests/test_enhanced_bayesian.py

# Test causal graph (with learnable)
python tests/test_causal_graph.py
```

All tests verify:
- Learnable modules are PyTorch nn.Modules
- Parameters are registered and require gradients
- Forward/backward passes work correctly
- Backward compatibility with fixed weights maintained

## Advantages

✅ **End-to-End Training**: Medical knowledge learned from data  
✅ **Domain Adaptation**: Weights adapt to specific populations  
✅ **Interpretability**: Learned weights can be inspected/compared to clinical values  
✅ **Minimal Overhead**: ~220 parameters vs. millions in neural nets  
✅ **Backward Compatible**: Fixed-weight mode still available  
✅ **Gradient-Based Optimization**: Leverages modern ML techniques  

## Limitations

⚠️ **Requires Training Data**: Need labeled patient-condition pairs  
⚠️ **Risk of Overfitting**: Small parameter set can overfit to training distribution  
⚠️ **Clinical Validation**: Learned weights should be validated by clinicians  
⚠️ **Initialization Matters**: Starts from clinical priors (not random)  

## Future Work

Potential extensions:
1. **Learnable Temporal Delays**: Make temporal delays trainable (currently fixed)
2. **Learnable Interaction Effects**: Make cross-condition interactions learnable
3. **Learnable Age/Severity Modifiers**: Make age and severity multipliers learnable
4. **Meta-Learning**: Learn to adapt weights quickly to new conditions
5. **Uncertainty Quantification**: Bayesian inference over learned weights
6. **Regularization**: L1/L2 penalties to prevent drift from clinical values

## Example Workflow

Complete example of using learnable modules:

```python
from phaita.training.adversarial_trainer import AdversarialTrainer
from phaita.data.synthetic_generator import generate_synthetic_dataset

# Generate synthetic training data
train_data = generate_synthetic_dataset(num_samples=1000)

# Create trainer with all learnable modules
trainer = AdversarialTrainer(
    use_pretrained_generator=True,
    use_pretrained_discriminator=True,
    use_learnable_bayesian=True,
    use_learnable_comorbidity=True,
    use_learnable_causality=True,
    device="cuda"
)

# Train
trainer.train(
    num_epochs=10,
    batch_size=8,
    real_dataset=train_data
)

# Inspect learned weights
if trainer.learnable_comorbidity:
    diabetes_effects = trainer.learnable_comorbidity.get_symptom_modifiers("diabetes")
    print("Learned diabetes effects:", diabetes_effects)

if trainer.learnable_causality:
    causal_edges = trainer.learnable_causality.get_causal_edges()
    print("Learned causal strengths:", causal_edges[:5])

# Save learned weights
import torch
if trainer.learnable_comorbidity:
    torch.save(trainer.learnable_comorbidity.state_dict(), "learned_comorbidity.pt")
if trainer.learnable_causality:
    torch.save(trainer.learnable_causality.state_dict(), "learned_causality.pt")
```

## References

- **Clinical Initialization**: YAML configs based on medical guidelines (GINA, GOLD, ADA, etc.)
- **Architecture**: Inspired by learnable graph neural networks and medical knowledge bases
- **Training**: Standard gradient descent with backpropagation through graphs

## Support

For questions or issues with learnable modules:
1. Check tests: `tests/test_learnable_modules.py`
2. See implementation: `phaita/models/learnable_comorbidity.py`, `phaita/models/learnable_causality.py`
3. Read docs: This guide + module docstrings
