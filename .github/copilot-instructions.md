# PHAITA Copilot Instructions

## Overview
PHAITA (Pre-Hospital AI Triage Algorithm) - medical triage research prototype using adversarial training (generator vs. discriminator). ~850KB, 28 Python files, Python 3.10+. Uses PyTorch, transformers (DeBERTa, Mistral 7B), GNNs, Bayesian networks. **Dual-mode:** full deep-learning OR lightweight template fallback. Research only - NOT clinical use.

## Build & Test Commands

**Install dependencies (ALWAYS DO FIRST):**
```bash
pip install -r requirements.txt  # ~2-5 min
pip install -e .  # optional editable install
```

**Tests (all are plain Python scripts, NO pytest needed):**
```bash
python test_basic.py              # FAST ~10s - data, Bayesian, config, synthetic
python test_enhanced_bayesian.py  # FAST ~10s - age/severity/rare presentations
python test_forum_scraping.py     # FAST ~10s - forum data, lay language
python test_integration.py        # SLOW ~2min - requires network, retries HuggingFace downloads
```
⚠️ Integration test attempts model downloads, retries ~60s, then falls back gracefully (expected).

**Demos:**
```bash
python simple_demo.py                              # No dependencies, instant
python cli.py demo --num-examples 5                # Uses template fallback
python cli.py generate --count 10 --output out.json
python cli.py diagnose --complaint "can't breathe"
```

**Network behavior:** Without access to huggingface.co, system retries ~30-60s per model, then uses template mode (NORMAL, HANDLED). Warnings are expected offline.

## Project Structure

```
phaita/                         # Core package
  data/
    icd_conditions.py           # 10 ICD-10 respiratory conditions
    forum_scraper.py            # Lay/medical term mapping
    synthetic_generator.py      # Batch generation
    preprocessing.py            # Text normalization
  models/
    discriminator.py            # DeBERTa + GNN classifier (~3.8M params)
    generator.py                # Mistral 7B / templates (512 params fallback)
    bayesian_network.py         # Symptom sampler
    enhanced_bayesian_network.py # Age/severity/rare logic
    gnn_module.py               # Symptom graph attention
    question_generator.py       # Clarifying questions
  training/
    adversarial_trainer.py      # G/D alternating optimization
  utils/
    config.py                   # YAML loader (config.yaml)
    metrics.py                  # Accuracy, diversity
    realism_scorer.py           # Transformer realism scoring
cli.py                          # Main interface (train/demo/generate/diagnose/challenge)
test_*.py                       # Test suites (4 files)
demo_*.py                       # Demos (3 files)
config.yaml                     # Model names, hyperparams, architecture
requirements.txt                # Dependencies (torch>=2.0, transformers>=4.35, etc.)
```

**Key exports (phaita/__init__.py):** RespiratoryConditions, SymptomGenerator, ComplaintGenerator, DiagnosisDiscriminator, BayesianSymptomNetwork, AdversarialTrainer, Config

**Data flow:** ICD-10 condition → BayesianSymptomNetwork samples symptoms → ComplaintGenerator (Mistral/templates) → DiagnosisDiscriminator (DeBERTa+GNN) predicts → AdversarialTrainer alternates updates

## Making Changes

**Test after changes:**
```bash
python test_basic.py              # Always run first
python test_enhanced_bayesian.py  # If Bayesian logic changed
python test_forum_scraping.py     # If data layer changed
# Skip test_integration.py unless testing full stack
```

**Key files for changes:**
- Generator: `phaita/models/generator.py`
- Discriminator: `phaita/models/discriminator.py`
- Conditions: `phaita/data/icd_conditions.py`
- Bayesian: `phaita/models/bayesian_network.py`, `enhanced_bayesian_network.py`

**Dependencies:** Edit BOTH `requirements.txt` AND `setup.py`, then `pip install -r requirements.txt`

## Critical Information

**Dual-mode operation:**
- Template mode: Fast, CPU, offline, 8 templates, deterministic (DEFAULT)
- Full stack: GPU (4GB+ VRAM), downloads ~7GB models, `use_pretrained=True`

**Automatic fallback:** Missing torch_geometric → MLP encoder; missing models → templates; missing transformers → simple text. **DON'T fix "failed to download" errors - expected offline.**

**Test expectations:** test_basic.py (4/4), test_enhanced_bayesian.py (all), test_forum_scraping.py (3/3), test_integration.py (may timeout Task 1).

**Python 3.10+** required. Check: `python --version`

**NO pytest/unittest** - plain Python scripts only.

**CPU/GPU:** Everything works CPU-only with templates. GPU optional for full stack. Auto-detects device.

## Documentation
- `README.md` - Quick start, CLI, API
- `PROJECT_SUMMARY.md` - Problem, solution, vision
- `IMPLEMENTATION_SUMMARY.md` - Module guide
- `IMPLEMENTATION_DETAILS.md` - DL architecture
- `DEEP_LEARNING_GUIDE.md` - GPU setup, troubleshooting
- `CHANGE_HISTORY.md` - Fixes, outstanding work

**Key facts:** 10 respiratory ICD-10 conditions, adversarial training, Akinator-style vision (WIP), recent fixes (generator ref bug, discriminator PyTorch compat, grammar, forum realism).

## Quick Reference

**Always:**
- ✅ `pip install -r requirements.txt` first
- ✅ `python test_basic.py` to verify (~10s)
- ✅ Use template mode for dev/test
- ✅ Expect HuggingFace retries offline (normal)
- ✅ Read docstrings for dual-mode behavior

**Never:**
- ❌ Add pytest/unittest
- ❌ Fix "can't connect to huggingface.co"
- ❌ Assume GPU
- ❌ Modify medical priors without review
- ❌ Add heavy dependencies

**Trust these instructions.** System designed for offline work with graceful degradation.
