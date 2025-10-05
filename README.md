# PHAITA

**Pre-Hospital AI Triage Algorithm**

PHAITA is a research prototype for medical triage that pits a language-model complaint generator against a diagnosis discriminator. The system explores how synthetic data, adversarial training, and medical knowledge graphs can improve robustness without using real patient data. The flagship deliverable is an Akinator-style conversational assistant that interviews patients, asks clarifying questions, and ultimately surfaces ten differential diagnoses with probabilities, red-flag warnings, and escalation guidance.

## Overview
- **Scope**: Ten respiratory conditions (ICD-10) with lay-language support.
- **Goal**: Stress-test diagnostic models with challenging, human-like complaints.
- **Status**: Production-ready deep-learning pipeline requiring GPU acceleration and transformer models.

📦 **[Installation Guide](INSTALLATION.md)** - Detailed instructions for all installation methods (CPU-only, GPU, development, etc.)  
⚡ **[Quick Reference](INSTALL_QUICK_REF.md)** - One-line install commands and quick decision tree

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

PHAITA supports modular installation based on your needs:

#### **Option 1: Full Installation (Recommended for Development)**
Install all dependencies including GPU features, development tools, and web scraping:
```bash
pip install -r requirements.txt
# or with setup.py
pip install -e .[all]
```

#### **Option 2: Minimal Installation (CPU-Only)**
Install only core dependencies for CPU-only operation:
```bash
pip install -r requirements-base.txt
# or with setup.py
pip install -e .
```

#### **Option 3: Custom Installation**
Install specific feature sets as needed:
```bash
# Base + GPU features (bitsandbytes, torch-geometric)
pip install -e .[gpu]

# Base + Development tools (pytest)
pip install -e .[dev]

# Base + Web scraping (praw, beautifulsoup4)
pip install -e .[scraping]
```

#### Core Dependencies (Always Required)
- **Python**: 3.10+ (3.12 recommended)
- **PyTorch**: 2.5.1
- **Transformers**: 4.46.0 (HuggingFace)
- **Plus**: datasets, accelerate, numpy, pandas, scikit-learn, networkx, scipy, matplotlib, seaborn, tqdm, pyyaml, safetensors, requests

#### Optional GPU Dependencies (requirements-gpu.txt)
- **bitsandbytes**: 0.44.1 (for 4-bit model quantization, **CUDA required**)
- **torch-geometric**: 2.6.1 (for Graph Neural Networks, **falls back to MLP without it**)

#### Optional Development Dependencies (requirements-dev.txt)
- **pytest**: >=7.0 (for running tests)

#### Optional Web Scraping Dependencies (requirements-scraping.txt)
- **praw**: >=7.7.0 (Reddit API for forum data collection)
- **beautifulsoup4**: >=4.12.0 (HTML parsing)

**Note**: GPU dependencies require CUDA. CPU-only environments should skip `requirements-gpu.txt` or use the base installation option.

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
| Training loop | AdversarialTrainer with curriculum, diversity, and unsupervised forum regularisation | Alternate generator/discriminator optimization on synthetic + forum-style text with masked unlabeled samples. |

Curriculum batches now expose a boolean mask so unlabeled forum complaints skip
cross-entropy updates. Instead, the discriminator minimises the entropy of its
predictions on those samples, encouraging confident assignments without
hallucinating random labels.

## Key Capabilities
- **Synthetic-first pipeline**: Generates complaints, question prompts, and labeled datasets without patient data.
- **Lay-language understanding**: Bidirectional mapping between medical and colloquial terms plus curated forum phrases.
- **Production-ready**: Requires real transformer models and GPU acceleration for consistent results.
- **Metrics and analysis**: Track diagnostic accuracy, diversity, realism, and failure cases from challenge evaluations.

## Getting Started

### Installation

Choose your installation method based on your environment and needs:

#### Full Installation (All Features)
```bash
# Install all dependencies (GPU, dev tools, scraping)
pip install -r requirements.txt

# Or use setup.py with all extras
pip install -e .[all]
```

#### Minimal Installation (CPU-Only, Core Features)
```bash
# Install only core dependencies
pip install -r requirements-base.txt

# Or use setup.py
pip install -e .
```

#### Custom Installation
```bash
# Core + GPU features only
pip install -e .[gpu]

# Core + Development tools only
pip install -e .[dev]

# Core + Web scraping only
pip install -e .[scraping]

# Mix and match
pip install -e .[gpu,dev]
```

### Quick Start with main.py

PHAITA now includes a simplified `main.py` entry point for common tasks:

```bash
# Run a quick demo (easiest way to get started)
python main.py demo

# Train the model
python main.py train --epochs 50

# Diagnose a patient complaint
python main.py diagnose --complaint "I can't breathe"

# Interactive diagnosis session
python main.py diagnose --interactive

# Interactive patient simulation
python main.py interactive

# Generate synthetic data
python main.py generate --count 10

# Access full CLI features
python main.py cli --help
```

### Advanced CLI Usage

For advanced features, use `cli.py` or `patient_cli.py` directly:

```bash
# Run an interactive demo with specific options
python demos/simple_demo.py

# Generate synthetic complaints with custom output
python cli.py generate --count 10 --output complaints.json

# Train with specific hyperparameters
python cli.py train --epochs 50 --batch-size 16 --lr 0.001

# Diagnose with detailed analysis
python cli.py diagnose --complaint "I can't catch my breath" --detailed

# Run challenge mode
python cli.py challenge --rare-cases 5 --show-failures
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

## Repository Structure
```
phaita/                # Core package (data, models, training, utils, conversation, triage)
├── data/              # Medical conditions, forum scraping, synthetic generation
├── models/            # Neural networks (generator, discriminator, Bayesian, GNN)
├── training/          # Adversarial trainer
├── utils/             # Config, metrics, model loader, realism scorer
├── conversation/      # Dialogue engine, conversation flow
└── triage/            # Diagnosis orchestrator, red-flags, escalation

docs/                  # Documentation
├── guides/            # Implementation and training guides (SOP)
├── modules/           # Module-specific documentation
├── updates/           # Consolidated update logs
├── architecture/      # Architecture documentation
└── features/          # Feature-specific guides

tests/                 # All test files (27 test scripts)
demos/                 # Demo scripts (10 demos)
config/                # YAML configuration files
scripts/               # Utility scripts (forum scraping, profiling)

main.py                # Centralized entry point for common tasks ⭐ NEW
cli.py                 # Command-line interface (advanced features)
patient_cli.py         # Web interface
config.yaml            # Main configuration file
```

cli.py                 # Command-line interface
patient_cli.py         # Web interface
config.yaml            # Main configuration file
```

## Testing

PHAITA includes **26 comprehensive test suites** (no pytest required - plain Python):

```bash
# Quick validation - core tests (~30 seconds)
python tests/test_basic.py                      # Core data, config, Bayesian
python tests/test_enhanced_bayesian.py          # Age/severity/rare presentations
python tests/test_forum_scraping.py             # Forum data augmentation
python tests/test_dialogue_engine.py            # Belief updating, information gain
python tests/test_diagnosis_orchestrator.py     # Ensemble, red-flags, escalation

# Integration tests - end-to-end workflows (~10 seconds)
python tests/test_conversation_flow.py          # Complete triage sessions
python tests/test_escalation_guidance.py        # Care routing validation
```

**Test coverage includes:**
- ✅ Core data layer and configuration
- ✅ Bayesian networks (basic + enhanced features)
- ✅ Multi-turn dialogue systems
- ✅ Diagnosis orchestration with red-flags
- ✅ Graph neural networks with causal edges
- ✅ Template generation systems
- ✅ Temporal symptom modeling
- ✅ Uncertainty quantification
- ✅ CLI interfaces and workflows

See **[docs/TESTING.md](docs/TESTING.md)** for the complete testing guide with all test files documented.

## Documentation Map

### Core Documentation (Start Here)
- **[docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)** – Complete navigation guide to all documentation
- **[docs/guides/SOP.md](docs/guides/SOP.md)** – Comprehensive Standard Operating Procedure (training, implementation, running)
- **[docs/TESTING.md](docs/TESTING.md)** – Complete testing guide with all 26 test files documented
- **[README.md](README.md)** – This file - quick start and overview

### Module Documentation
- **[docs/modules/DATA_MODULE.md](docs/modules/DATA_MODULE.md)** – Data layer, ICD conditions, forum scraping, red-flags
- **[docs/modules/MODELS_MODULE.md](docs/modules/MODELS_MODULE.md)** – Neural networks, Bayesian networks, GNN, generator/discriminator
- **[docs/modules/CONVERSATION_MODULE.md](docs/modules/CONVERSATION_MODULE.md)** – Dialogue engine, belief updating, conversation flow
- **[docs/modules/TRIAGE_MODULE.md](docs/modules/TRIAGE_MODULE.md)** – Diagnosis orchestration, red-flags, escalation guidance
- **[docs/modules/IMPLEMENTATION_SUMMARY.md](docs/modules/IMPLEMENTATION_SUMMARY.md)** – High-level architecture overview
- **[docs/modules/IMPLEMENTATION_DETAILS.md](docs/modules/IMPLEMENTATION_DETAILS.md)** – Deep-learning technical details

### Guides and References
- **[docs/updates/UPDATE_LOG.md](docs/updates/UPDATE_LOG.md)** – Consolidated update history, fixes, and verifications
- **[DEEP_LEARNING_GUIDE.md](DEEP_LEARNING_GUIDE.md)** – GPU setup and troubleshooting
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** – Problem statement, solution outline, and roadmap
- **[CHANGE_HISTORY.md](CHANGE_HISTORY.md)** – Project evolution and outstanding work

## License & Disclaimer
PHAITA is released under the MIT License. The project is for research and educational purposes only and must not be used for real-world medical decision making without regulatory approval.
