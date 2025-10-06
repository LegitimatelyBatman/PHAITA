# PHAITA Copilot Instructions

## Overview
PHAITA (Pre-Hospital AI Triage Algorithm) - medical triage research prototype using adversarial training (generator vs. discriminator). ~850KB, 28 Python files, Python 3.10+. Uses PyTorch, transformers (DeBERTa, Mistral 7B), GNNs, Bayesian networks. **ML-first architecture:** Attempts machine learning by default, automatically falls back to lightweight templates if ML unavailable. Research only - NOT clinical use.

## Build & Test Commands

**Install dependencies (ALWAYS DO FIRST):**
```bash
pip install -r requirements.txt  # ~2-5 min
pip install -e .  # optional editable install
```

**Tests (all are plain Python scripts, NO pytest needed):**
```bash
# Core tests (always run these first)
python tests/test_basic.py              # FAST ~10s - data, Bayesian, config, synthetic
python tests/test_enhanced_bayesian.py  # FAST ~10s - age/severity/rare/comorbidity
python tests/test_forum_scraping.py     # FAST ~10s - forum data, lay language

# Dialogue and conversation tests
python tests/test_dialogue_engine.py            # FAST ~5s - belief updating, info gain
python tests/test_diagnosis_orchestrator.py     # FAST ~3s - ensemble, red-flags, escalation
python tests/test_conversation_flow.py          # FAST ~5s - end-to-end triage sessions
python tests/test_escalation_guidance.py        # FAST ~3s - care routing validation

# Model and architecture tests
python tests/test_causal_graph.py               # FAST ~5s - GNN causal edges
python tests/test_template_diversity.py         # FAST ~5s - template generation
python tests/test_temporal_modeling.py          # FAST ~10s - temporal symptom progression
python tests/test_uncertainty.py                # FAST ~5s - uncertainty quantification
python tests/test_gnn_performance.py            # FAST ~10s - GNN benchmarks

# CLI and workflow tests
python tests/test_conversation_engine.py        # FAST ~5s - conversation engine logic
python tests/test_cli_challenge_command.py      # FAST ~2s - challenge CLI
python tests/test_cli_triage_workflow.py        # FAST ~5s - diagnose CLI workflow

# Integration tests
python tests/test_integration.py                # SLOW ~2min - requires network, downloads models

# Note: Additional test files exist for specific features (model loader, patient CLI, etc.)
# See docs/TESTING.md for complete documentation of all 26 test files
```
⚠️ Integration test attempts model downloads, retries ~60s, then falls back gracefully (expected).

**Demos:**
```bash
python demos/simple_demo.py                         # No dependencies, instant
python cli.py demo --num-examples 5                 # ML-first, falls back if needed
python cli.py generate --count 10 --output out.json
python cli.py diagnose --complaint "can't breathe"
```

**Network behavior:** Without access to huggingface.co, system retries ~30-60s per model, prints warnings, then automatically falls back to template mode (NORMAL, EXPECTED). System continues working seamlessly.

## Project Structure

```
phaita/                         # Core package
  data/                         # Medical data and scraping
    icd_conditions.py           # 10 ICD-10 respiratory conditions
    forum_scraper.py            # Lay/medical term mapping
    synthetic_generator.py      # Batch generation
    preprocessing.py            # Text normalization
    red_flags.py                # Emergency criteria
  models/                       # Neural networks and ML
    discriminator.py            # DeBERTa + GNN classifier (~3.8M params)
    generator.py                # Mistral 7B / templates (512 params fallback)
    bayesian_network.py         # Symptom sampler
    enhanced_bayesian_network.py # Age/severity/rare logic
    gnn_module.py               # Symptom graph attention
    question_generator.py       # Clarifying questions
  conversation/                 # Dialogue management
    dialogue_engine.py          # Belief updating, info gain
    engine.py                   # Conversation flow control
  triage/                       # Diagnosis and escalation
    diagnosis_orchestrator.py   # Ranked diagnoses with red-flags
    diagnosis.py                # Individual diagnosis generation
    info_sheet.py               # Patient information sheets
  training/                     # Model training
    adversarial_trainer.py      # G/D alternating optimization
  utils/                        # Utilities
    config.py                   # YAML loader (config.yaml)
    metrics.py                  # Accuracy, diversity
    realism_scorer.py           # Transformer realism scoring
    model_loader.py             # Model download with retry logic

tests/                          # All test files (26 files)
demos/                          # Demo scripts (10 files)
docs/                           # Documentation
  guides/                       # SOP and training guides
  modules/                      # Module-specific documentation
  updates/                      # Consolidated update logs
  architecture/                 # Architecture documentation
  features/                     # Feature-specific guides

config/                         # YAML configuration files
scripts/                        # Utility scripts
cli.py                          # Main CLI interface
patient_cli.py                  # Web interface
```

**Key exports (phaita/__init__.py):** RespiratoryConditions, SymptomGenerator, ComplaintGenerator, DiagnosisDiscriminator, BayesianSymptomNetwork, AdversarialTrainer, Config

**Data flow:** ICD-10 condition → BayesianSymptomNetwork samples symptoms → ComplaintGenerator (Mistral/templates) → DiagnosisDiscriminator (DeBERTa+GNN) predicts → AdversarialTrainer alternates updates

## Making Changes

**Test after changes:**
```bash
# Always run core tests first
python tests/test_basic.py              # Core functionality
python tests/test_enhanced_bayesian.py  # If Bayesian logic changed
python tests/test_forum_scraping.py     # If data layer changed

# Run feature-specific tests based on changes
python tests/test_dialogue_engine.py            # If dialogue system changed
python tests/test_diagnosis_orchestrator.py     # If diagnosis/red-flags changed
python tests/test_conversation_flow.py          # If conversation flow changed
python tests/test_causal_graph.py               # If GNN/graph changed
python tests/test_template_diversity.py         # If template system changed

# See docs/TESTING.md for complete test guide
# Skip test_integration.py unless testing full stack
```

**Key files for changes:**
- Generator: `phaita/models/generator.py`
- Discriminator: `phaita/models/discriminator.py`
- Conditions: `phaita/data/icd_conditions.py`
- Bayesian: `phaita/models/bayesian_network.py`, `enhanced_bayesian_network.py`

**Dependencies:** Edit BOTH `requirements.txt` AND `setup.py`, then `pip install -r requirements.txt`

## Critical Information

**ML-first operation with automatic fallback:**
- **Primary mode:** ML (DeBERTa+GNN, Mistral-7B) - Attempted first by default
- **Fallback mode:** Templates/lightweight - Automatic when ML unavailable
- **Behavior:** System tries ML, warns if unavailable, falls back seamlessly
- **No configuration needed:** Just use default parameters (ML-first)

**Automatic fallback triggers:**
- Missing torch_geometric → MLP encoder
- Missing/unavailable models → Templates/keyword matching
- Missing transformers → Simple text processing
- Offline/no internet → Template mode after retries
- Insufficient memory → Lightweight mode

**Expected behavior offline:** Model download retries (~30-60s), prints warnings explaining the issue, automatically falls back to templates. **DON'T fix "failed to download" errors - this is NORMAL and HANDLED.**

**Test expectations:** test_basic.py (4/4), test_enhanced_bayesian.py (6/6), test_forum_scraping.py (3/3), test_dialogue_engine.py (all pass), test_diagnosis_orchestrator.py (11/11), test_conversation_flow.py (5/5), test_escalation_guidance.py (6/6). See docs/TESTING.md for all 26 test files.

**Python 3.10+** required. Check: `python --version`

**NO pytest/unittest** - plain Python scripts only.

**CPU/GPU:** Everything works CPU-only with automatic fallback. GPU optional for best quality. Auto-detects device.

## Documentation
- `docs/DOCUMENTATION_INDEX.md` - Complete navigation guide to all documentation
- `docs/guides/SOP.md` - Comprehensive Standard Operating Procedure (training, implementation, running)
- `docs/TESTING.md` - Comprehensive testing guide (all 26 test files)
- `docs/TESTING_MULTI_TURN_DIALOGUES.md` - Dialogue integration tests
- `docs/modules/DATA_MODULE.md` - Data layer documentation
- `docs/modules/MODELS_MODULE.md` - Neural networks documentation (updated with ML-first info)
- `docs/modules/CONVERSATION_MODULE.md` - Dialogue engine documentation
- `docs/modules/TRIAGE_MODULE.md` - Diagnosis and red-flags documentation
- `docs/modules/IMPLEMENTATION_SUMMARY.md` - High-level architecture
- `docs/modules/IMPLEMENTATION_DETAILS.md` - Deep-learning details
- `docs/updates/UPDATE_LOG.md` - Consolidated update history
- `README.md` - Quick start, CLI, API (updated with ML-first info)
- `PROJECT_SUMMARY.md` - Problem, solution, vision
- `DEEP_LEARNING_GUIDE.md` - GPU setup, troubleshooting
- `CHANGE_HISTORY.md` - Fixes, outstanding work

**Key facts:** 10 respiratory ICD-10 conditions, adversarial training, Akinator-style vision (WIP), ML-first architecture with automatic fallback.

## Quick Reference

**Always:**
- ✅ `pip install -r requirements.txt` first
- ✅ `python tests/test_basic.py` to verify (~10s)
- ✅ Use default parameters (ML-first with automatic fallback)
- ✅ Expect HuggingFace retries offline (normal, handled automatically)
- ✅ Read warnings - they explain what's happening and why
- ✅ Trust the fallback - system works seamlessly in both modes

**Never:**
- ❌ Add pytest/unittest
- ❌ Fix "can't connect to huggingface.co" (normal offline behavior, handled automatically)
- ❌ Assume GPU
- ❌ Modify medical priors without review
- ❌ Add heavy dependencies
- ❌ Force template mode with use_pretrained=False (let system try ML first)

**Trust these instructions.** System designed for ML-first operation with seamless offline fallback.
