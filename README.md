# PHAITA

**Pre-Hospital AI Triage Algorithm**

PHAITA is a research prototype for medical triage that pits a language-model complaint generator against a diagnosis discriminator. The system explores how synthetic data, adversarial training, and medical knowledge graphs can improve robustness without using real patient data. The flagship deliverable is an Akinator-style conversational assistant that interviews patients, asks clarifying questions, and ultimately surfaces ten differential diagnoses with probabilities, red-flag warnings, and escalation guidance.

## Overview
- **Scope**: Ten respiratory conditions (ICD-10) with lay-language support.
- **Goal**: Stress-test diagnostic models with challenging, human-like complaints.
- **Status**: End-to-end demo pipeline with mock clinical workflows and configurable depth (template-only through full deep-learning stack).

## Architecture Snapshot
| Stage | Components | Purpose |
|-------|------------|---------|
| Complaint generation | Bayesian symptom sampler → (optional) Mistral 7B → template fallback | Produce varied patient narratives for a target condition. |
| Diagnosis | DeBERTa-based encoder + symptom Graph Neural Network | Predict condition and flag synthetic complaints. |
| Training loop | AdversarialTrainer with curriculum and diversity losses | Alternate generator/discriminator optimisation on synthetic + forum-style text. |

## Key Capabilities
- **Synthetic-first pipeline**: Generates complaints, question prompts, and labeled datasets without patient data.
- **Lay-language understanding**: Bidirectional mapping between medical and colloquial terms plus curated forum phrases.
- **Configurable depth**: Run lightweight demos, or enable quantized LLM and full PyTorch training when resources allow.
- **Metrics and analysis**: Track diagnostic accuracy, diversity, realism, and failure cases from challenge evaluations.

## Getting Started
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: editable install
pip install -e .
```

### CLI Recipes
```bash
# Run an interactive demo
python cli.py demo --num-examples 5

# Generate synthetic complaints
python cli.py generate --count 10 --output complaints.json

# Train the adversarial loop
python cli.py train --epochs 50 --batch-size 16

# Diagnose a custom complaint
python cli.py diagnose --complaint "I can't catch my breath"
```

### Python API Example
```python
from phaita import AdversarialTrainer
from phaita.models import ComplaintGenerator, DiagnosisDiscriminator

generator = ComplaintGenerator()
discriminator = DiagnosisDiscriminator()
trainer = AdversarialTrainer(generator=generator, discriminator=discriminator)

symptoms = ["shortness_of_breath", "wheezing"]
complaint = generator.generate_complaint(symptoms, "J45.9")
print(complaint)

predictions = discriminator.predict_diagnosis([complaint])
print(predictions[0])
```

## Repository Guide
```
phaita/                Core package (data, models, training, utils)
cli.py                 Command-line interface for demos and tools
demo_*.py              Ready-to-run showcase scripts
test_*.py              Unit and integration tests
*.md                   Focused documentation (architecture, implementation, research notes)
```

## Documentation Map
- **PROJECT_SUMMARY.md** – Problem statement, solution outline, and roadmap.
- **IMPLEMENTATION_SUMMARY.md** – High-level tour of major modules.
- **IMPLEMENTATION_DETAILS.md** – Deep-learning upgrade highlights and references.
- **DEEP_LEARNING_GUIDE.md** – Practical guidance for enabling the full model stack.
- **CHANGE_HISTORY.md** – Consolidated audit notes, historical fixes, and outstanding work.

## License & Disclaimer
PHAITA is released under the MIT License. The project is for research and educational purposes only and must not be used for real-world medical decision making without regulatory approval.
