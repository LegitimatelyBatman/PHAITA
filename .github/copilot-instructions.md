# PHAITA Copilot Instructions

## Project Overview

PHAITA (Pre-Hospital AI Triage Algorithm) is a **research prototype** for medical triage using adversarial training. The system features a language-model complaint generator vs. a diagnosis discriminator to improve robustness without real patient data. **Size:** ~850KB, **28 Python files**, **Python 3.10+** required.

**Key Technologies:**
- PyTorch with transformers (DeBERTa, Mistral 7B)
- Graph Neural Networks (torch-geometric)
- Adversarial training with Bayesian networks
- Dual-mode operation: full deep-learning stack OR lightweight template fallback

**Critical:** This is for research/education only - NOT for clinical use.

## Build & Test Commands

### Environment Setup

**ALWAYS install dependencies first:**
```bash
pip install -r requirements.txt
```
This installs PyTorch, transformers, torch-geometric, and related dependencies. Installation takes ~2-5 minutes.

**Optional editable install:**
```bash
pip install -e .
```

### Testing

**Run all basic tests (FAST - 10-15 seconds):**
```bash
python test_basic.py
```
Tests data layer, Bayesian network, config system, and synthetic generation WITHOUT requiring network access or model downloads.

**Run enhanced Bayesian tests (FAST - 5 seconds):**
```bash
python test_enhanced_bayesian.py
```

**Run forum scraping tests (FAST - 5 seconds):**
```bash
python test_forum_scraping.py
```

**Run integration tests (SLOW - 2+ minutes, requires network):**
```bash
python test_integration.py
```
⚠️ **WARNING:** This test attempts to download models from HuggingFace. Without network access, it will retry multiple times and timeout after ~60-120 seconds. The system falls back to template mode gracefully, but the test takes time. Skip this test if working offline.

**All tests are executable Python scripts** - NO pytest, unittest, or special test runners required. Simply run `python test_*.py`.

### Demo Commands

**Simple demo (NO dependencies, instant):**
```bash
python simple_demo.py
```
Shows respiratory conditions and basic data layer - works without torch installed.

**Deep learning demo (requires dependencies):**
```bash
python demo_deep_learning.py
```
⚠️ Attempts to load pretrained models from HuggingFace. Without network, falls back to template mode with warnings (~60 seconds of retries).

**CLI demos (recommended - uses template fallback):**
```bash
python cli.py demo --num-examples 5
python cli.py generate --count 10 --output complaints.json
python cli.py diagnose --complaint "I can't breathe"
```
These commands gracefully fall back to template-based generation when models can't be downloaded.

### Known Issues & Workarounds

**Network Access:**
- The system tries to download models from HuggingFace (huggingface.co)
- Without network: retries for ~30-60 seconds per model, then falls back to template mode
- This is EXPECTED and HANDLED - the fallback mode works correctly
- Warnings like "Could not load LLM" are normal when offline

**Template vs. Full Stack:**
- Template mode: Fast, CPU-only, no model downloads, deterministic
- Full stack mode: Requires GPU (4GB+ VRAM), downloads ~7GB models, much slower
- All CLI commands work in template mode - this is the default safe path

## Project Architecture

### Directory Structure

```
/
├── phaita/                     # Core package
│   ├── data/                   # Data layer
│   │   ├── icd_conditions.py   # ICD-10 respiratory conditions (10 conditions)
│   │   ├── forum_scraper.py    # Mock forum posts & lay/medical term mapping
│   │   ├── synthetic_generator.py  # Batch complaint generation helpers
│   │   └── preprocessing.py    # Text normalization & tokenization
│   ├── models/                 # Model components
│   │   ├── discriminator.py    # DeBERTa + GNN diagnosis classifier
│   │   ├── generator.py        # Mistral 7B / template complaint generator
│   │   ├── bayesian_network.py # Baseline symptom sampler
│   │   ├── enhanced_bayesian_network.py  # Age/severity/rare presentation logic
│   │   ├── gnn_module.py       # Symptom graph attention network
│   │   └── question_generator.py  # Clarifying question prompts
│   ├── training/               # Training loop
│   │   └── adversarial_trainer.py  # Alternating G/D optimization
│   └── utils/                  # Utilities
│       ├── config.py           # YAML config loader (config.yaml)
│       ├── metrics.py          # Accuracy, diversity metrics
│       └── realism_scorer.py   # Transformer-based realism scoring
├── cli.py                      # Main CLI (train, demo, generate, diagnose, challenge)
├── simple_demo.py              # No-dependency demo script
├── demo_deep_learning.py       # Full model stack demo
├── demo_fixes.py               # Bug fix demonstrations
├── test_basic.py               # Basic tests (FAST)
├── test_integration.py         # Integration tests (SLOW)
├── test_enhanced_bayesian.py   # Enhanced Bayesian tests
├── test_forum_scraping.py      # Forum data tests
├── config.yaml                 # Default configuration
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── *.md                        # Documentation
```

### Configuration Files

**config.yaml** - Main configuration:
- Model names (DeBERTa, Mistral)
- Training hyperparameters (epochs, batch size, learning rates)
- GNN architecture (hidden dims, layers)
- Data parameters (conditions, symptoms per condition)

**requirements.txt** - All dependencies including:
- torch>=2.0.0
- transformers>=4.35.0
- torch-geometric>=2.3.0
- Plus 15+ other packages

**setup.py** - Package metadata and install configuration

### Key Entry Points

**cli.py** - Main interface with subcommands:
- `train`: Train adversarial models
- `demo`: Show system capabilities
- `generate`: Create synthetic complaints
- `diagnose`: Test discriminator on input
- `challenge`: Run adversarial challenge mode

**phaita/__init__.py** - Package exports:
- RespiratoryConditions
- SymptomGenerator, ComplaintGenerator
- DiagnosisDiscriminator
- BayesianSymptomNetwork
- AdversarialTrainer
- Config

### Data Flow

1. **Condition Selection:** ICD-10 code from RespiratoryConditions (10 respiratory conditions)
2. **Symptom Sampling:** BayesianSymptomNetwork samples symptoms based on priors
3. **Complaint Generation:** ComplaintGenerator creates patient narrative (Mistral 7B or templates)
4. **Diagnosis:** DiagnosisDiscriminator predicts condition (DeBERTa + symptom graph)
5. **Training:** AdversarialTrainer alternates generator/discriminator updates

### Operating Modes

**Template Mode (default fallback):**
- No model downloads required
- Uses 8 grammatically-correct templates
- Deterministic symptom selection
- Fast, CPU-only
- Works offline

**Full Stack Mode (optional):**
- Requires GPU (4GB+ VRAM recommended)
- Downloads DeBERTa (~500MB) and Mistral 7B (~7GB in 4-bit quantized form)
- Real transformer-based generation
- Enable with `use_pretrained=True` in code or via config

## Common Tasks

### Making Code Changes

**Running tests after changes:**
```bash
python test_basic.py              # Fast smoke tests
python test_enhanced_bayesian.py  # If changing Bayesian logic
python test_forum_scraping.py     # If changing data layer
# Skip test_integration.py unless testing full stack
```

**Testing CLI changes:**
```bash
python cli.py --help
python cli.py demo --num-examples 2
python cli.py generate --count 2 --output /tmp/test.json
```

**Modifying models:**
- Generator: `phaita/models/generator.py`
- Discriminator: `phaita/models/discriminator.py`
- Test with template mode first (no `use_pretrained` flag)

**Modifying data:**
- Conditions: `phaita/data/icd_conditions.py`
- Bayesian priors: `phaita/models/bayesian_network.py`
- Enhanced logic: `phaita/models/enhanced_bayesian_network.py`

### Adding Dependencies

Edit `requirements.txt` AND `setup.py` to keep them in sync. Then:
```bash
pip install -r requirements.txt
```

### Running in Offline/No-Network Mode

**All these work offline:**
- `python simple_demo.py`
- `python test_basic.py`
- `python test_enhanced_bayesian.py`
- `python test_forum_scraping.py`
- `python cli.py generate --count 10` (uses templates)

**These require network or fail gracefully:**
- `python test_integration.py` (retries, then template fallback)
- `python cli.py demo` (with warnings)
- `python demo_deep_learning.py` (with warnings)

## Important Implementation Details

### Fallback Mechanisms

The codebase has **automatic graceful degradation**:
- Missing torch_geometric → MLP-based graph encoder
- Missing pretrained models → Template-based generation
- Missing transformers → Simple text processing

**DO NOT** try to fix "failed to download model" errors - this is expected behavior when offline. The system handles it.

### Test Expectations

- `test_basic.py`: 4/4 tests pass (always)
- `test_enhanced_bayesian.py`: All tests pass (always)
- `test_forum_scraping.py`: 3/3 tests pass (always)
- `test_integration.py`: May timeout on Task 1 (realism scorer needs network), Tasks 2-4 should pass

### Python Version

**Python 3.10+** required (documented in DEEP_LEARNING_GUIDE.md). The code uses modern type hints and transformers features that require recent Python.

Check version: `python --version`

### GPU/CPU Notes

- **CPU mode:** Everything works, uses templates
- **GPU mode:** Optional, for full DeBERTa + Mistral stack
- No CUDA required for basic development/testing
- Models auto-detect device via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`

## Critical Don'ts

1. **DON'T** add pytest/unittest infrastructure - tests are plain Python scripts
2. **DON'T** try to fix "can't connect to huggingface.co" - it's handled
3. **DON'T** assume GPU availability - always test CPU/template path
4. **DON'T** modify medical condition priors without clinical review (research project disclaimer)
5. **DON'T** add heavyweight dependencies - project prioritizes dual-mode operation

## Documentation References

**Read these for context:**
- `README.md` - Quick start, CLI examples, API overview
- `PROJECT_SUMMARY.md` - Problem statement, solution approach, vision
- `IMPLEMENTATION_SUMMARY.md` - Module-by-module guide
- `IMPLEMENTATION_DETAILS.md` - Deep learning architecture highlights
- `DEEP_LEARNING_GUIDE.md` - GPU setup, troubleshooting, full stack mode
- `CHANGE_HISTORY.md` - Recent fixes, outstanding work, audit findings

**Key facts from docs:**
- 10 respiratory conditions (ICD-10)
- Adversarial training: generator vs. discriminator
- Akinator-style triage conversation (vision - not fully implemented)
- Recent critical fixes: generator reference bug, discriminator PyTorch compat, grammar corrections, forum data realism

## Summary Checklist

When working on PHAITA, always:
- ✅ Install dependencies first: `pip install -r requirements.txt`
- ✅ Run `python test_basic.py` to verify environment (10-15 seconds)
- ✅ Test CLI commands with small examples first
- ✅ Expect HuggingFace download retries and warnings (normal when offline)
- ✅ Use template mode for development/testing (fast, reliable)
- ✅ Read docstrings in code - they explain dual-mode behavior
- ✅ Check config.yaml for model names and hyperparameters
- ✅ Verify changes don't break template fallback mode

**Trust these instructions.** Only search/explore if information is incomplete or contradicts observed behavior. The system is designed to work offline with graceful degradation - embrace the fallback mechanisms.
