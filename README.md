# PHAITA
**Pre-Hospital AI Triage Algorithm**

An adversarial training system for medical triage using Bayesian Networks, Mistral 7B, and DeBERTa + GNN for intelligent diagnosis from patient language.

## 🏥 Overview

PHAITA implements a novel adversarial training approach for medical triage:

- **Generator**: Bayesian Network + Quantized Mistral 7B converts medical conditions to realistic patient complaints
- **Discriminator**: DeBERTa + Graph Neural Network diagnoses conditions from patient language  
- **Training**: Adversarial training with diversity loss prevents repetition and improves generalization
- **Focus**: 10 respiratory conditions with handling of lay terminology

## 🌟 Key Features

- ✅ **No Real Patient Data Required**: Fully synthetic training pipeline
- ✅ **Adversarial Training**: Generator creates challenging cases, discriminator learns robust diagnosis
- ✅ **Diversity Loss**: Prevents repetitive complaints and encourages varied expression
- ✅ **Medical Knowledge Graph**: GNN leverages symptom relationships for better diagnosis
- ✅ **Quantized Models**: Efficient inference with 4-bit quantized Mistral 7B
- ✅ **Lay Terminology**: Handles non-medical patient language effectively

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LegitimatelyBatman/PHAITA.git
cd PHAITA

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```bash
# Run a quick demo
python cli.py demo --num-examples 5

# Generate synthetic patient data
python cli.py generate --count 10 --output examples.json

# Train the adversarial model
python cli.py train --epochs 50 --batch-size 16
```

### Python API

```python
from phaita import AdversarialTrainer, RespiratoryConditions
from phaita.models import ComplaintGenerator, DiagnosisDiscriminator

# Initialize components
trainer = AdversarialTrainer()
generator = ComplaintGenerator()
discriminator = DiagnosisDiscriminator()

# Generate synthetic complaint
symptoms = ["shortness_of_breath", "wheezing", "chest_tightness"]
complaint = generator.generate_complaint(symptoms, "J45.9")
print(f"Patient says: '{complaint}'")

# Diagnose from complaint
diagnosis = discriminator.predict_diagnosis([complaint])
print(f"AI diagnosis: {diagnosis[0]}")
```

## 🏗️ Architecture

### Generator Pipeline
```
ICD-10 Code → Bayesian Network → Symptoms → Mistral 7B → Patient Complaint
```

### Discriminator Pipeline  
```
Patient Complaint → DeBERTa Encoder → Text Features
Medical Knowledge → Graph Neural Network → Graph Features
                    ↓
            Fusion Layer → Diagnosis + Authenticity Score
```

### Adversarial Training Loop
```
1. Generator creates synthetic complaints
2. Discriminator tries to classify condition + detect fake complaints  
3. Generator optimizes to fool discriminator (+ diversity loss)
4. Discriminator optimizes to correctly classify + detect fakes
5. Repeat until convergence
```

## 📊 Respiratory Conditions

The system handles 10 respiratory conditions based on ICD-10 codes:

| Code | Condition | Example Symptoms |
|------|-----------|------------------|
| J44.1 | COPD with Acute Exacerbation | Shortness of breath, wheezing, chronic cough |
| J45.9 | Asthma, unspecified | Wheezing, chest tightness, difficulty breathing |
| J18.9 | Pneumonia, unspecified | Fever, chills, chest pain, cough |
| J20.9 | Acute Bronchitis | Persistent cough, chest discomfort, fatigue |
| J12.9 | Viral Pneumonia | Dry cough, fever, headache, muscle aches |
| J13 | Bacterial Pneumonia | Sudden fever, productive cough, chest pain |
| J21.9 | Acute Bronchiolitis | Wheezing, rapid breathing, difficulty feeding |
| J84.10 | Pulmonary Fibrosis | Progressive dyspnea, dry cough, fatigue |
| J44.0 | COPD with Infection | Increased cough, purulent sputum, fever |
| R06.02 | Acute Dyspnea | Sudden shortness of breath, anxiety |

## 🔧 Configuration

Customize training and model parameters in `config.yaml`:

```yaml
model:
  deberta_model: "microsoft/deberta-base"
  mistral_model: "mistralai/Mistral-7B-Instruct-v0.1"
  gnn_hidden_dim: 128
  use_quantization: true

training:
  num_epochs: 100
  batch_size: 16
  diversity_weight: 0.1
  generator_lr: 2.0e-5
  discriminator_lr: 1.0e-4
```

## 📈 Training Metrics

The system tracks several metrics during training:

- **Diagnosis Accuracy**: Correct condition classification
- **Lexical Diversity**: Vocabulary variety in generated complaints  
- **Semantic Diversity**: Embedding-based semantic variation
- **Discriminator Loss**: Real vs. fake complaint detection
- **Generator Loss**: Adversarial + diversity objectives

## 🛠️ Advanced Usage

### Custom Training

```python
from phaita import AdversarialTrainer, Config

# Load custom configuration
config = Config.from_yaml("custom_config.yaml")

# Initialize trainer with custom parameters
trainer = AdversarialTrainer(
    diversity_weight=0.2,  # Higher diversity emphasis
    generator_lr=1e-5,     # Lower learning rate
)

# Train with custom settings
history = trainer.train(
    num_epochs=200,
    batch_size=32,
    eval_interval=5
)
```

### Generating Diverse Complaints

```python
from phaita.models import ComplaintGenerator

generator = ComplaintGenerator()
symptoms = ["fever", "cough", "chest_pain"]

# Generate multiple diverse variants
complaints = generator.generate_diverse_complaints(
    symptoms, "J18.9", num_variants=5
)

for i, complaint in enumerate(complaints):
    print(f"Variant {i+1}: {complaint}")
```

## 🔬 Research Applications

PHAITA can be used for:

- **Medical AI Training**: Synthetic data generation for rare conditions
- **Triage System Development**: Automated patient prioritization  
- **Clinical Decision Support**: Diagnostic assistance tools
- **Medical Education**: Training simulations with diverse patient presentations
- **Adversarial Robustness**: Testing diagnostic models against challenging cases

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code style and standards
- Testing requirements  
- Documentation updates
- Feature requests and bug reports

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Citation

If you use PHAITA in your research, please cite:

```bibtex
@software{phaita2024,
  title={PHAITA: Pre-Hospital AI Triage Algorithm with Adversarial Training},
  author={PHAITA Team},
  year={2024},
  url={https://github.com/LegitimatelyBatman/PHAITA}
}
```

## ⚠️ Disclaimer

This system is for research and educational purposes only. It should not be used for actual medical diagnosis or treatment decisions without proper validation and regulatory approval.
