# Models Module Documentation

**Location:** `phaita/models/`

## Overview

The models module contains all neural network architectures and machine learning components, including the discriminator, generator, Bayesian networks, GNN modules, and question generation.

**ML-First Architecture:** All models default to attempting machine learning first, with automatic graceful fallback to lightweight CPU-friendly alternatives if ML models are unavailable (e.g., offline, insufficient memory, missing dependencies).

## Components

### 1. Generator (`generator.py`)

**Purpose:** Generate natural-language patient complaints from medical conditions and symptoms.

**Key Classes:**
- `ComplaintGenerator` - Main generator interface
- `SymptomGenerator` - Generate symptom lists

**Dual-Mode Architecture with Automatic Fallback:**

#### Mode 1: Machine Learning (Primary)
- **Model:** Mistral-7B instruction-tuned
- **Quantization:** 4-bit with bitsandbytes
- **VRAM:** ~4GB required (or 16GB+ RAM for CPU)
- **Quality:** High-quality natural narratives
- **Activation:** Automatically attempted by default

#### Mode 2: Template Fallback (Automatic)
- **Model:** Enhanced template system
- **Parameters:** 512 learnable parameters
- **Memory:** <100MB
- **Quality:** Good grammar, diverse templates
- **Activation:** Automatic fallback when ML unavailable

**Usage:**
```python
from phaita.models.generator import ComplaintGenerator

# Initialize (ML-first with automatic fallback)
generator = ComplaintGenerator()  # Attempts Mistral-7B, falls back to templates if needed

# The system will:
# 1. Try to load Mistral-7B with retries
# 2. Print warning if ML unavailable
# 3. Automatically fall back to template mode
# 4. Continue working seamlessly

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
- `use_pretrained`: Whether to attempt ML (default: True - ML-first)
- `quantization`: "4bit", "8bit", or None
- `max_length`: Max output tokens
- `temperature`: Sampling temperature

---

### 2. Discriminator (`discriminator.py`)

**Purpose:** Classify patient complaints into diagnoses with confidence scores.

**Key Class:** `DiagnosisDiscriminator`

**Dual-Mode Architecture with Automatic Fallback:**

#### Mode 1: Machine Learning (Primary)
- **Text Encoder:** DeBERTa-v3-base (184M params)
- **Graph Module:** Symptom GNN with attention (100K params)
- **Fusion Layer:** Multi-head fusion (50K params)
- **Output Heads:** 10-way classifier + confidence
- **Total Parameters:** ~3.8M
- **Activation:** Automatically attempted by default

#### Mode 2: Lightweight Fallback (Automatic)
- **Text Encoder:** Keyword-based feature extraction
- **Parameters:** <10K
- **Memory:** <50MB
- **Quality:** Good accuracy on clear symptoms
- **Activation:** Automatic fallback when ML unavailable

**Features:**
- Multi-task learning (diagnosis + confidence)
- Graph-structured symptom reasoning (ML mode)
- Attention over symptom relationships (ML mode)
- Gradient-based training (ML mode)
- Keyword matching (fallback mode)

**Usage:**
```python
from phaita.models.discriminator import DiagnosisDiscriminator

# Initialize (ML-first with automatic fallback)
discriminator = DiagnosisDiscriminator()  # Attempts DeBERTa+GNN, falls back to lightweight if needed

# The system will:
# 1. Try to load DeBERTa and GNN models with retries
# 2. Print warning if ML unavailable
# 3. Automatically fall back to lightweight mode
# 4. Continue working seamlessly

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

**Key Classes:** 
- `BayesianSymptomNetwork` - Standard network with fixed probabilities
- `LearnableBayesianSymptomNetwork` - Neural network with learnable weights

**Features:**
- Conditional probability tables (CPTs)
- Prior and posterior probabilities
- Symptom independence assumptions
- Evidence propagation
- Learnable weights via gradient descent (learnable variant)

**Standard Usage:**
```python
from phaita.models.bayesian_network import BayesianSymptomNetwork

network = BayesianSymptomNetwork()

# Get symptom probability given condition
prob = network.get_symptom_probability(
    condition_code="J45.9",
    symptom="shortness_of_breath"
)
print(f"P(shortness_of_breath | asthma) = {prob:.2f}")

# Sample symptoms for a condition
symptoms = network.sample_symptoms(
    condition_code="J45.9",
    num_symptoms=5
)
print(f"Sampled symptoms: {symptoms}")
```

**Learnable Network Usage:**
```python
from phaita.models.bayesian_network import LearnableBayesianSymptomNetwork
import torch

# Initialize learnable network
network = LearnableBayesianSymptomNetwork(device="cuda")

# Network is a PyTorch nn.Module with learnable parameters
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

# Get probabilities (these are learnable)
primary_prob, severity_prob = network.get_probabilities("J45.9")
print(f"Primary: {primary_prob:.3f}, Severity: {severity_prob:.3f}")

# Sample symptoms (using learned probabilities)
symptoms = network.sample_symptoms("J45.9", num_symptoms=4)

# Train with medical accuracy loss
from phaita.utils.medical_loss import MedicalAccuracyLoss
loss_fn = MedicalAccuracyLoss()

sampled_symptoms = [symptoms]
condition_codes = ["J45.9"]
loss = loss_fn(sampled_symptoms, condition_codes, network)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Learnable Architecture:**
- **Primary symptom logit:** nn.Parameter initialized to logit(0.8)
- **Severity symptom logit:** nn.Parameter initialized to logit(0.4)
- **Per-condition weights:** nn.Parameter of shape [num_conditions, 2]
- **Total parameters:** ~20 for 10 conditions
- **Memory:** <1KB
- **Training:** Integrated with AdversarialTrainer

**Mathematical Foundation:**
```
Standard: P(symptom | condition) = fixed_probability

Learnable: P(symptom | condition) = σ(logit + condition_adjustment)
where σ is the sigmoid function and adjustments are learned
```

**Key Differences:**
| Feature | BayesianSymptomNetwork | LearnableBayesianSymptomNetwork |
|---------|------------------------|--------------------------------|
| Probabilities | Fixed (0.8, 0.4) | Learned via gradient descent |
| PyTorch Integration | No | Yes (nn.Module) |
| Training | N/A | MedicalAccuracyLoss |
| Memory | <100KB | <1KB parameters |
| GPU Support | N/A | Yes (optional) |

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
- ✅ **Use defaults** - Models are now ML-first by default with automatic fallback
- ✅ **Trust the fallback** - System gracefully degrades when ML unavailable
- ✅ **Enable enhanced Bayesian features** for better realism
- ✅ **Profile memory** before deploying full models in production
- ✅ **Read warnings** - System informs you when falling back and why

### DON'T:
- ❌ **Force template mode** - Let the system try ML first (default behavior)
- ❌ **Ignore warnings** - They explain what's happening and how to enable ML
- ❌ **Skip normalization** in Bayesian network
- ❌ **Train without diversity loss** (causes mode collapse)
- ❌ **Assume mode** - Check `template_mode` / `use_pretrained` attributes to verify which mode is active

### ML-First Migration Guide

**Old approach (deprecated):**
```python
# Explicitly forcing fallback mode
gen = ComplaintGenerator(use_pretrained=False)  # ❌ Don't do this
disc = DiagnosisDiscriminator(use_pretrained=False)  # ❌ Don't do this
```

**New approach (recommended):**
```python
# Let system try ML first, fall back automatically
gen = ComplaintGenerator()  # ✅ ML-first with automatic fallback
disc = DiagnosisDiscriminator()  # ✅ ML-first with automatic fallback

# Check which mode is active
if gen.template_mode:
    print("Generator using template mode (fallback)")
else:
    print("Generator using ML mode (Mistral-7B)")

if not disc.use_pretrained:
    print("Discriminator using lightweight mode (fallback)")
else:
    print("Discriminator using ML mode (DeBERTa+GNN)")
```

---

## Related Documentation

- [DATA_MODULE.md](DATA_MODULE.md) - Data layer integration
- [TRAINING_GUIDE.md](../guides/TRAINING_GUIDE.md) - Training procedures
- [DEEP_LEARNING_GUIDE.md](../../DEEP_LEARNING_GUIDE.md) - GPU setup
- [UPDATE_LOG.md](../updates/UPDATE_LOG.md) - Model updates

---

**Last Updated:** 2025-01-03
