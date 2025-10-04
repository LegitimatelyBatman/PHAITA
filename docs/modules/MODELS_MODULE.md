# Models Module Documentation

**Location:** `phaita/models/`

## Overview

The models module contains all neural network architectures and machine learning components, including the discriminator, generator, Bayesian networks, GNN modules, and question generation.

## Components

### 1. Generator (`generator.py`)

**Purpose:** Generate natural-language patient complaints from medical conditions and symptoms.

**Key Classes:**
- `ComplaintGenerator` - Main generator interface
- `SymptomGenerator` - Generate symptom lists

**Dual-Mode Architecture:**

#### Mode 1: Full Deep Learning (GPU)
- **Model:** Mistral-7B instruction-tuned
- **Quantization:** 4-bit with bitsandbytes
- **VRAM:** ~4GB required
- **Quality:** High-quality natural narratives

#### Mode 2: Template Fallback (CPU)
- **Model:** Enhanced template system
- **Parameters:** 512 learnable parameters
- **Memory:** <100MB
- **Quality:** Good grammar, diverse templates

**Usage:**
```python
from phaita.models.generator import ComplaintGenerator

# Initialize (auto-detects GPU/CPU)
generator = ComplaintGenerator(
    use_pretrained=True,  # Try Mistral-7B, fall back to templates
    device="auto"
)

# Generate complaint
complaint = generator.generate_complaint(
    condition_code="J45.9",
    symptoms=["shortness_of_breath", "wheezing", "chest_tightness"],
    severity="moderate"
)
print(complaint)
# Output: "I've been having trouble breathing for the past few days. 
#          My chest feels tight and I'm wheezing when I breathe."
```

**Configuration:**
- `model_name`: Transformer model (default: "mistralai/Mistral-7B-Instruct-v0.2")
- `use_pretrained`: Use transformer vs templates
- `quantization`: "4bit", "8bit", or None
- `max_length`: Max output tokens
- `temperature`: Sampling temperature

---

### 2. Discriminator (`discriminator.py`)

**Purpose:** Classify patient complaints into diagnoses with confidence scores.

**Key Class:** `DiagnosisDiscriminator`

**Architecture:**
- **Text Encoder:** DeBERTa-v3-base (184M params)
- **Graph Module:** Symptom GNN with attention (100K params)
- **Fusion Layer:** Multi-head fusion (50K params)
- **Output Heads:** 10-way classifier + confidence

**Total Parameters:** ~3.8M

**Features:**
- Multi-task learning (diagnosis + confidence)
- Graph-structured symptom reasoning
- Attention over symptom relationships
- Gradient-based training

**Usage:**
```python
from phaita.models.discriminator import DiagnosisDiscriminator

# Initialize
discriminator = DiagnosisDiscriminator(
    num_conditions=10,
    use_pretrained=True,
    device="auto"
)

# Diagnose complaint
complaint = "I can't breathe and my chest is tight"
diagnosis = discriminator.diagnose(complaint)

print(f"Top diagnosis: {diagnosis['condition']}")
print(f"Confidence: {diagnosis['confidence']:.2%}")
print(f"All probabilities: {diagnosis['probabilities']}")
```

**Outputs:**
- `condition`: Top predicted ICD-10 code
- `confidence`: Confidence score (0-1)
- `probabilities`: Distribution over all conditions
- `features`: Intermediate representations

---

### 3. Discriminator Lite (`discriminator_lite.py`)

**Purpose:** Lightweight discriminator for CPU-only or resource-constrained deployments.

**Key Class:** `DiscriminatorLite`

**Architecture:**
- **Text Encoder:** TF-IDF + MLP (10K params)
- **No GNN:** Simple feedforward
- **Fast inference:** <100ms on CPU

**Usage:**
```python
from phaita.models.discriminator_lite import DiscriminatorLite

discriminator = DiscriminatorLite(num_conditions=10)
diagnosis = discriminator.diagnose("I have chest pain")
```

**Use Cases:**
- CPU-only deployment
- Mobile/edge devices
- Fast prototyping
- Testing without GPU

---

### 4. Bayesian Network (`bayesian_network.py`)

**Purpose:** Probabilistic symptom modeling using Bayesian networks.

**Key Class:** `BayesianSymptomNetwork`

**Features:**
- Conditional probability tables (CPTs)
- Prior and posterior probabilities
- Symptom independence assumptions
- Evidence propagation

**Usage:**
```python
from phaita.models.bayesian_network import BayesianSymptomNetwork

network = BayesianSymptomNetwork()

# Get symptom probability given condition
prob = network.get_symptom_probability(
    condition_code="J45.9",
    symptom="shortness_of_breath"
)
print(f"P(shortness_of_breath | asthma) = {prob:.2f}")

# Update beliefs with evidence
evidence = {"shortness_of_breath": True, "fever": False}
posteriors = network.compute_posteriors(evidence)
print(f"P(asthma | evidence) = {posteriors['J45.9']:.2f}")
```

**Mathematical Foundation:**
```
P(condition | symptoms) ∝ P(symptoms | condition) × P(condition)

P(symptoms | condition) = ∏ P(symptom_i | condition)
```

---

### 5. Enhanced Bayesian Network (`enhanced_bayesian_network.py`)

**Purpose:** Extended Bayesian network with age, severity, comorbidities, and rare presentations.

**Key Class:** `EnhancedBayesianSymptomNetwork`

**Additional Features:**
- Age-based modulation (pediatric, adult, geriatric)
- Severity adjustment (mild, moderate, severe)
- Comorbidity effects (diabetes, hypertension, etc.)
- Rare presentation generation
- Priority-based symptom selection

**Usage:**
```python
from phaita.models.enhanced_bayesian_network import EnhancedBayesianSymptomNetwork

network = EnhancedBayesianSymptomNetwork()

# Sample symptoms with age and severity
symptoms = network.sample_symptoms(
    condition_code="J45.9",
    age=65,  # Geriatric patient
    severity="severe",
    comorbidities=["diabetes", "hypertension"]
)
print(f"Symptoms: {symptoms}")

# Adjust probabilities for age
prob = network.get_symptom_probability(
    condition_code="J45.9",
    symptom="shortness_of_breath",
    age=5  # Pediatric presentation differs
)
```

**Age Modulation:**
- **Pediatric (<18):** Adjusted symptom probabilities
- **Adult (18-65):** Standard probabilities
- **Geriatric (>65):** Increased atypical presentations

**Severity Modulation:**
- **Mild:** Fewer, less severe symptoms
- **Moderate:** Typical presentation
- **Severe:** More symptoms, higher intensities

---

### 6. GNN Module (`gnn_module.py`)

**Purpose:** Graph neural network for symptom relationship modeling.

**Key Class:** `SymptomGNN`

**Architecture:**
- Graph Attention Network (GAT)
- 3 layers with residual connections
- Edge features for causal relationships
- Node features for symptom embeddings

**Features:**
- Learn symptom-symptom relationships
- Incorporate medical causality
- Attention mechanism for relevance
- Integrate with discriminator

**Usage:**
```python
from phaita.models.gnn_module import SymptomGNN

gnn = SymptomGNN(
    input_dim=768,  # DeBERTa output
    hidden_dim=256,
    num_layers=3
)

# Forward pass
node_features = ...  # Symptom embeddings
edge_index = ...     # Symptom graph edges
output = gnn(node_features, edge_index)
```

**Graph Structure:**
- **Nodes:** Symptoms
- **Edges:** Causal or correlational relationships
- **Features:** Symptom embeddings + medical knowledge

---

### 7. Question Generator (`question_generator.py`)

**Purpose:** Generate clarifying questions for multi-turn dialogue.

**Key Class:** `QuestionGenerator`

**Features:**
- Information gain-based question selection
- Avoid repetition
- Context-aware generation
- Ranked question suggestions

**Usage:**
```python
from phaita.models.question_generator import QuestionGenerator

generator = QuestionGenerator()

# Generate questions based on current beliefs
questions = generator.generate_questions(
    current_beliefs={"J45.9": 0.6, "J44.0": 0.3, "J18.9": 0.1},
    asked_symptoms=["shortness_of_breath", "wheezing"],
    max_questions=5
)

for q in questions:
    print(f"Q: {q['text']}")
    print(f"   Information gain: {q['info_gain']:.3f}")
```

**Question Selection Criteria:**
1. **Information Gain:** Maximize entropy reduction
2. **Discriminative Power:** Separate top conditions
3. **Clinical Relevance:** Follow medical reasoning
4. **Avoid Repetition:** Don't ask twice

---

### 8. Temporal Module (`temporal_module.py`)

**Purpose:** Model temporal symptom progression and trajectories.

**Key Class:** `TemporalSymptomModel`

**Features:**
- Symptom onset timing
- Progression patterns
- Acute vs. chronic differentiation
- Time-aware predictions

**Usage:**
```python
from phaita.models.temporal_module import TemporalSymptomModel

model = TemporalSymptomModel()

# Model symptom progression
trajectory = model.predict_trajectory(
    condition_code="J45.9",
    initial_symptoms=["mild_wheezing"],
    timespan_days=7
)

print("Day 1:", trajectory[0])
print("Day 7:", trajectory[6])
```

**Applications:**
- Predict symptom evolution
- Distinguish acute vs. chronic
- Estimate disease progression
- Inform follow-up timing

---

## Model Training

### Adversarial Training (`phaita/training/adversarial_trainer.py`)

**Purpose:** Train generator and discriminator in adversarial loop.

**Key Class:** `AdversarialTrainer`

**Training Loop:**
1. **Generator step:** Generate synthetic complaints
2. **Discriminator step:** Classify real vs. synthetic
3. **Update both:** Alternating gradient descent
4. **Compute losses:** Adversarial + diversity + realism

**Usage:**
```python
from phaita.training.adversarial_trainer import AdversarialTrainer

trainer = AdversarialTrainer(
    generator=generator,
    discriminator=discriminator,
    learning_rate=1e-4
)

# Train
trainer.train(
    num_epochs=50,
    batch_size=16,
    real_data=real_complaints
)
```

**Losses:**
- **Adversarial Loss:** Generator fools discriminator
- **Diversity Loss:** Prevent mode collapse
- **Realism Loss:** Match real complaint distribution

---

## Configuration

### Model Configuration (`config.yaml`)

```yaml
models:
  generator:
    model_name: "mistralai/Mistral-7B-Instruct-v0.2"
    use_pretrained: true
    quantization: "4bit"
    temperature: 0.7
    max_length: 256
  
  discriminator:
    model_name: "microsoft/deberta-v3-base"
    use_pretrained: true
    hidden_dim: 768
    num_heads: 8
    dropout: 0.1
  
  bayesian:
    use_enhanced: true
    enable_age_modulation: true
    enable_severity_modulation: true
    enable_comorbidity_effects: true
```

---

## Testing

**Test Files:**
- `tests/test_basic.py` - Bayesian network basics
- `tests/test_enhanced_bayesian.py` - Enhanced features
- `tests/test_discriminator_lite.py` - Lite discriminator
- `tests/test_model_loader.py` - Model loading
- `tests/test_gnn_performance.py` - GNN benchmarks
- `tests/test_temporal_modeling.py` - Temporal features

**Run Tests:**
```bash
python tests/test_basic.py
python tests/test_enhanced_bayesian.py
pytest tests/test_model_loader.py
```

---

## Performance Benchmarks

### Generator
| Mode | Memory | Inference Time | Quality Score |
|------|--------|----------------|---------------|
| Mistral-7B (4-bit) | ~4GB | ~500ms | 9.2/10 |
| Templates | <100MB | ~10ms | 7.5/10 |

### Discriminator
| Mode | Memory | Inference Time | Accuracy |
|------|--------|----------------|----------|
| DeBERTa + GNN | ~1.5GB | ~200ms | 87% |
| Lite | <50MB | ~50ms | 78% |

---

## Best Practices

### DO:
- ✅ Use `use_pretrained=True` with graceful fallback
- ✅ Enable enhanced Bayesian features for better realism
- ✅ Profile memory before deploying full models
- ✅ Use discriminator lite for CPU-only deployments

### DON'T:
- ❌ Load full models without checking VRAM
- ❌ Ignore device compatibility warnings
- ❌ Skip normalization in Bayesian network
- ❌ Train without diversity loss (causes mode collapse)

---

## Related Documentation

- [DATA_MODULE.md](DATA_MODULE.md) - Data layer integration
- [TRAINING_GUIDE.md](../guides/TRAINING_GUIDE.md) - Training procedures
- [DEEP_LEARNING_GUIDE.md](../../DEEP_LEARNING_GUIDE.md) - GPU setup
- [UPDATE_LOG.md](../updates/UPDATE_LOG.md) - Model updates

---

**Last Updated:** 2025-01-03
