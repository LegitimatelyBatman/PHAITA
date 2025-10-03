# PHAITA

**Pre-Hospital AI Triage Algorithm**

PHAITA is a research prototype for medical triage that pits a language-model complaint generator against a diagnosis discriminator. The system explores how synthetic data, adversarial training, and medical knowledge graphs can improve robustness without using real patient data. The flagship deliverable is an Akinator-style conversational assistant that interviews patients, asks clarifying questions, and ultimately surfaces ten differential diagnoses with probabilities, red-flag warnings, and escalation guidance.

## Overview
- **Scope**: Ten respiratory conditions (ICD-10) with lay-language support.
- **Goal**: Stress-test diagnostic models with challenging, human-like complaints.
- **Status**: Research prototype with an optional deep-learning stack; template-based fallbacks keep demos and tests CPU-friendly.

## System Requirements

### Hardware Requirements

| Scenario | Hardware Notes |
|----------|----------------|
| **Template / smoke tests** | Runs entirely on CPU. A modern laptop (8 GB RAM) is sufficient for demos, unit tests, and deterministic complaint templates. |
| **Full deep-learning stack** | CUDA-capable GPU with **≥4 GB VRAM** (8 GB+ recommended for training) to host quantised Mistral 7B and DeBERTa. Expect ~10 GB of storage for model weights and an internet connection on first run. |

### Software Dependencies

- **Python**: 3.10+ (3.12 recommended)
- **PyTorch**: 2.5.1 (with CUDA 11.8+ for GPU acceleration)
- **Transformers**: 4.46.0 (HuggingFace)
- **bitsandbytes**: 0.44.1 (only needed when enabling 4-bit quantisation)
- **torch-geometric**: 2.6.1 (required for the graph-enhanced discriminator)

The full list lives in `requirements.txt`. Lightweight usage paths import only the Bayesian simulators and configuration helpers, so most tests run even when the transformer/GNN extras are missing.

### Models Used

The following pretrained models are downloaded on demand when the deep-learning mode is enabled:

- **Mistral-7B-Instruct-v0.2** (~7 GB): Complaint and question generation
- **microsoft/deberta-v3-base** (~440 MB): Text encoding for diagnosis
- **Bio_ClinicalBERT** or **bert-base-uncased** (~420 MB): Realism scoring
- **GPT-2** (~500 MB): Perplexity evaluation

**Note**: Template fallbacks remain available. Pass `use_pretrained=False` to `ComplaintGenerator` for deterministic text and stick to the Bayesian layers when resources are limited.

## Architecture Snapshot
| Stage | Components | Purpose |
|-------|------------|---------|
| Complaint generation | Bayesian symptom sampler → Mistral-7B-Instruct (4-bit quantized) | Produce varied patient narratives for a target condition. |
| Diagnosis | DeBERTa-v3-base encoder + Graph Attention Network (torch-geometric) | Predict condition and assess complaint realism. |
| Training loop | AdversarialTrainer with curriculum, diversity, and unsupervised forum regularisation | Alternate generator/discriminator optimization on synthetic + forum-style text with masked unlabeled samples. |

Curriculum batches now expose a boolean mask so unlabeled forum complaints skip
cross-entropy updates. Instead, the discriminator minimises the entropy of its
predictions on those samples, encouraging confident assignments without
hallucinating random labels.

## Key Capabilities
- **Synthetic-first pipeline**: Generates complaints, question prompts, and labeled datasets without patient data.
- **Lay-language understanding**: Bidirectional mapping between medical and colloquial terms plus curated forum phrases.
- **Hybrid-friendly**: Deep-learning upgrades rely on real transformer models and GPUs, while the Bayesian/template path stays portable.
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

### Editing respiratory condition definitions

Respiratory conditions, symptoms, severity indicators, and lay-language vocabularies
now live in [`config/respiratory_conditions.yaml`](config/respiratory_conditions.yaml).
Clinicians can edit this file (or point the `PHAITA_RESPIRATORY_CONFIG`
environment variable at an alternative path) without touching the Python code.

```yaml
J45.9:
  name: Asthma
  symptoms:
    - wheezing
    - shortness_of_breath
  severity_indicators:
    - unable_to_speak
  lay_terms:
    - "can't breathe"
    - tight chest
  description: Chronic inflammatory airway disease with episodic symptoms
```

After editing the file, long-running services can hot-reload the catalogue:

```python
from phaita.data import RespiratoryConditions

# Reload from disk (uses PHAITA_RESPIRATORY_CONFIG if set)
RespiratoryConditions.reload()
```

`BayesianSymptomNetwork`, `SymptomGenerator`, and `ComplaintGenerator` subscribe to
these reload events automatically, so freshly authored conditions become available
immediately. Forum scraping and curriculum-learning utilities consume the same
vocabulary via `RespiratoryConditions.get_vocabulary()`, ensuring the NLP complaint
generator is trained on the exact symptom and lay-language set defined by physicians.

### Forum scraping
Real forum data is now harvested on-demand via `scripts/scrape_forums.py`. The
Reddit client uses OAuth credentials which must be supplied via environment
variables or command-line arguments in your own wrapper scripts.

```bash
# Required for Reddit scraping
export REDDIT_CLIENT_ID="..."
export REDDIT_CLIENT_SECRET="..."
export REDDIT_USER_AGENT="phaita-dev/0.1"

# Optional: authenticate as a specific user for private subreddit access
export REDDIT_USERNAME="..."
export REDDIT_PASSWORD="..."

# Scrape both Reddit and Patient.info forums and persist to forum_data/
python scripts/scrape_forums.py --source all --max-posts 50

# Scrape Reddit only into a custom cache directory
python scripts/scrape_forums.py --source reddit --cache-dir .cache/forums
```

The scraper overwrites cached JSON exports each run so downstream components
always consume fresh data. Patient.info scraping relies on standard HTTP
requests and respects a modest rate-limit by default.

### Python API Example
```python
from phaita import AdversarialTrainer
from phaita.models import ComplaintGenerator, DiagnosisDiscriminator, SymptomGenerator

# Initialize models (GPU recommended for LLM mode; templates stay CPU-friendly)
symptom_gen = SymptomGenerator()
generator = ComplaintGenerator(use_pretrained=True, use_4bit=True)
discriminator = DiagnosisDiscriminator(use_pretrained=True)
trainer = AdversarialTrainer(generator=generator, discriminator=discriminator)

# Generate and diagnose
presentation = symptom_gen.generate_symptoms("J45.9")
presentation = generator.generate_complaint(presentation=presentation)
print(presentation.complaint_text)

predictions = discriminator.predict_diagnosis([presentation.complaint_text], top_k=3)
for candidate in predictions[0]:
    print(candidate["condition_code"], candidate["probability"])
```

**Note**: `ComplaintGenerator` still exposes a deterministic template mode via `use_pretrained=False`. `DiagnosisDiscriminator` requires pretrained weights because its fallback encoder has been removed.

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
