# PHAITA

**Pre-Hospital AI Triage Algorithm**

PHAITA is a research prototype for medical triage that pits a language-model complaint generator against a diagnosis discriminator. The system explores how synthetic data, adversarial training, and medical knowledge graphs can improve robustness without using real patient data. The flagship deliverable is an Akinator-style conversational assistant that interviews patients, asks clarifying questions, and ultimately surfaces ten differential diagnoses with probabilities, red-flag warnings, and escalation guidance.

## Overview
- **Scope**: Ten respiratory conditions (ICD-10) with lay-language support.
- **Goal**: Stress-test diagnostic models with challenging, human-like complaints.
- **Status**: Production-ready deep-learning pipeline requiring GPU acceleration and transformer models.

## System Requirements

### Hardware Requirements
PHAITA requires significant computational resources for the full deep-learning stack:

- **GPU**: CUDA-capable GPU with **4GB+ VRAM** (8GB+ recommended for training)
  - Required for 4-bit quantization with bitsandbytes
  - CPU-only mode available but significantly slower (pass `use_4bit=False`)
- **RAM**: 16GB+ system memory recommended
- **Storage**: ~10GB for model weights (downloaded from HuggingFace Hub on first run)
- **Network**: Internet connection required for initial model downloads

### Software Dependencies
Exact versions are required for compatibility:

- **Python**: 3.10+ (3.12 recommended)
- **PyTorch**: 2.5.1 (with CUDA 11.8+ for GPU support)
- **Transformers**: 4.46.0 (HuggingFace)
- **bitsandbytes**: 0.44.1 (for 4-bit model quantization, CUDA required)
- **torch-geometric**: 2.6.1 (for Graph Neural Networks)

See `requirements.txt` for complete dependency list.

### Models Used
The following models are automatically downloaded from HuggingFace Hub:

- **Mistral-7B-Instruct-v0.2** (~7GB): Complaint and question generation
- **microsoft/deberta-v3-base** (~440MB): Text encoding for diagnosis
- **Bio_ClinicalBERT** or **bert-base-uncased** (~420MB): Realism scoring
- **GPT-2** (~500MB): Perplexity evaluation

**Note**: All models are now **required**. Template-based fallbacks have been removed to ensure consistent behavior and quality.

## Architecture Snapshot
| Stage | Components | Purpose |
|-------|------------|---------|
| Complaint generation | Bayesian symptom sampler → Mistral-7B-Instruct (4-bit quantized) | Produce varied patient narratives for a target condition. |
| Diagnosis | DeBERTa-v3-base encoder + Graph Attention Network (torch-geometric) | Predict condition and assess complaint realism. |
| Training loop | AdversarialTrainer with curriculum and diversity losses | Alternate generator/discriminator optimization on synthetic + forum-style text. |

## Key Capabilities
- **Synthetic-first pipeline**: Generates complaints, question prompts, and labeled datasets without patient data.
- **Lay-language understanding**: Bidirectional mapping between medical and colloquial terms plus curated forum phrases.
- **Production-ready**: Requires real transformer models and GPU acceleration for consistent results.
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

# Initialize models (requires GPU with 4GB+ VRAM or CPU with use_4bit=False)
symptom_gen = SymptomGenerator()
generator = ComplaintGenerator(use_pretrained=True, use_4bit=True)  # Required: use_pretrained=True
discriminator = DiagnosisDiscriminator(use_pretrained=True)  # Required: use_pretrained=True
trainer = AdversarialTrainer(generator=generator, discriminator=discriminator)

# Generate and diagnose
presentation = symptom_gen.generate_symptoms("J45.9")
presentation = generator.generate_complaint(presentation=presentation)
print(presentation.complaint_text)

predictions = discriminator.predict_diagnosis([presentation.complaint_text], top_k=3)
for candidate in predictions[0]:
    print(candidate["condition_code"], candidate["probability"])
```

**Note**: All models now require `use_pretrained=True` (the default). Attempting to use `use_pretrained=False` will raise a `ValueError`.

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
