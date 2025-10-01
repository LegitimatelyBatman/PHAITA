# PHAITA - Copilot Coding Agent Instructions

## Repository Overview

**PHAITA (Pre-Hospital AI Triage Algorithm)** is a medical triage system using adversarial training. It combines Bayesian Networks, Mistral 7B LLM (with 4-bit quantization), DeBERTa encoder, and Graph Neural Networks to generate realistic patient complaints and diagnose respiratory conditions from patient language.

- **Size**: 28 Python files, ~5,000 lines of code
- **Languages**: Python 3.8+
- **Frameworks**: PyTorch 2.0+, Transformers 4.35+, torch-geometric 2.3+
- **Target**: Medical AI research and synthetic data generation
- **Scope**: 10 respiratory conditions (ICD-10 codes: J45.9, J18.9, J44.9, J06.9, J20.9, J81.0, J93.0, J15.9, J12.9, J21.9)

## Build & Environment Setup

### Installation (Required Steps)

**ALWAYS run these commands in order for a clean setup:**

```bash
# 1. Install dependencies (~2-3 minutes, downloads ~2GB)
pip install -r requirements.txt

# 2. Install package in development mode (required for imports to work)
pip install -e .
```

### Python & Runtime Versions
- **Python**: 3.8+ (tested with 3.12.3)
- **pip**: 24.0+
- **PyTorch**: 2.0.0+ (installed automatically with CUDA 12.8 support)
- **Transformers**: 4.35.0+

### Key Dependencies
- `torch>=2.0.0` - Core deep learning framework
- `transformers>=4.35.0` - Pretrained models (DeBERTa, Mistral)
- `bitsandbytes>=0.39.0` - 4-bit quantization for Mistral
- `torch-geometric>=2.3.0` - Graph Neural Networks (has MLP fallback if unavailable)
- `networkx>=3.0` - Medical knowledge graph
- `scikit-learn>=1.3.0`, `numpy>=1.24.0`, `pandas>=2.0.0` - Data processing

**Note**: HuggingFace models (Mistral, DeBERTa) require internet access for first download (~4GB for Mistral). The system has template-based fallbacks that work offline.

## Testing & Validation

### Test Commands (Always run in this order)

**1. Basic Tests** (~10 seconds, no model downloads):
```bash
python test_basic.py
```
- Tests: Data layer, Bayesian network, config system, synthetic generation
- Expected: 4/4 tests passed
- **This is the fastest validation step - run first**

**2. Enhanced Bayesian Tests** (~10 seconds):
```bash
python test_enhanced_bayesian.py
```
- Tests: Age/severity modifiers, rare presentations, evidence sources
- Expected: All tests passed

**3. Forum Scraping Tests** (~10 seconds):
```bash
python test_forum_scraping.py
```
- Tests: Forum scraper, lay language mapper, data augmentation
- Expected: 3/3 tests passed

**4. Integration Tests** (~5+ minutes, attempts model downloads):
```bash
python test_integration.py
```
- Tests: End-to-end training loop, PyTorch compatibility
- **WARNING**: Attempts to download Mistral-7B (~4GB) and DeBERTa models
- **Falls back to template mode if downloads fail**
- Expected: 5/5 tests passed (or graceful fallback)

### Running Tests Without Internet
If HuggingFace access fails, models automatically fall back to template/CPU mode:
- Generator: Uses 8 grammatical templates instead of Mistral LLM
- Discriminator: Uses random predictions instead of DeBERTa
- **Tests still pass** with fallback behavior

### CLI Commands

**Demo** (~15-30 seconds with fallback):
```bash
python cli.py demo --num-examples 3
```
- Shows synthetic patient complaint generation
- Attempts LLM mode first, falls back to templates

**Generate Data** (~5-10 seconds with fallback):
```bash
python cli.py generate --count 10 --output examples.json
```
- Creates synthetic patient complaints
- Saves to JSON file

**Simple Demo** (No dependencies, ~5 seconds):
```bash
python simple_demo.py
```
- Lightweight demo using only data layer
- No model downloads required

### Test Infrastructure
- **No pytest/unittest**: Uses custom test runners with `main()` functions
- **Run tests directly**: `python test_<name>.py`
- **Exit codes**: 0 = success, 1 = failure

## Project Layout & Architecture

### Directory Structure
```
PHAITA/
├── phaita/                      # Main package
│   ├── data/                    # Medical data & processing
│   │   ├── icd_conditions.py    # 10 respiratory conditions (ICD-10)
│   │   ├── forum_scraper.py     # Lay language collection
│   │   └── preprocessing.py     # Data pipeline
│   ├── models/                  # Neural network models
│   │   ├── generator.py         # Mistral-7B complaint generator (+ templates)
│   │   ├── discriminator.py     # DeBERTa + GNN discriminator
│   │   ├── bayesian_network.py  # Symptom probability network
│   │   ├── enhanced_bayesian_network.py  # Age/severity modifiers
│   │   ├── gnn_module.py        # Graph Neural Network
│   │   └── question_generator.py # Triage question generation
│   ├── training/                # Training loops
│   │   └── adversarial_trainer.py  # GAN-style training
│   └── utils/                   # Configuration & metrics
│       ├── config.py            # YAML config loading
│       ├── metrics.py           # Diversity & accuracy metrics
│       └── realism_scorer.py    # Complaint authenticity scoring
├── cli.py                       # Command-line interface
├── test_basic.py               # Basic tests (fastest)
├── test_enhanced_bayesian.py   # Bayesian network tests
├── test_forum_scraping.py      # Forum scraping tests
├── test_integration.py         # Full integration tests (slowest)
├── simple_demo.py              # Lightweight demo
├── demo_deep_learning.py       # Deep learning demo
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
└── config.yaml                 # Default configuration
```

### Configuration Files

**config.yaml** - Model and training parameters:
```yaml
model:
  deberta_model: "microsoft/deberta-base"
  mistral_model: "mistralai/Mistral-7B-Instruct-v0.1"
  gnn_hidden_dim: 128
  use_quantization: true

training:
  num_epochs: 100
  batch_size: 16
  generator_lr: 2.0e-5
  discriminator_lr: 1.0e-4
  diversity_weight: 0.1
```

**setup.py** - Package configuration (use `pip install -e .` to install)

**requirements.txt** - Dependencies (use `pip install -r requirements.txt`)

**.gitignore** - Excludes model weights (*.pth, *.bin, *.safetensors), logs, temp files

### Key Source Files

**phaita/data/icd_conditions.py** (9,397 bytes):
- `RespiratoryConditions` class with 10 conditions
- Symptoms, severity indicators, lay terminology
- Methods: `get_condition_by_code()`, `get_symptoms_for_condition()`, `get_lay_terms_for_condition()`

**phaita/models/generator.py** (23,637 bytes):
- `ComplaintGenerator(nn.Module)` - Mistral-7B with 4-bit quantization
- Template fallback: 8 grammatical templates with 512 learnable parameters
- Usage: `generator.generate_complaint(symptoms, icd_code)`
- Memory: ~3.5GB VRAM (4-bit) or <1MB (template mode)

**phaita/models/discriminator.py** (16,837 bytes):
- `DiagnosisDiscriminator(nn.Module)` - DeBERTa encoder + GNN
- PyTorch-compatible: `.to(device)`, `.train()`, `.eval()`, `.parameters()`
- Returns: `{'diagnosis_logits': tensor, 'discriminator_scores': tensor}`

**phaita/training/adversarial_trainer.py** (29,716 bytes):
- `AdversarialTrainer` class - GAN-style training loop
- Combines generator and discriminator
- Diversity loss to prevent repetitive complaints

**cli.py** (22,727 bytes):
- Commands: `train`, `demo`, `generate`, `diagnose`, `challenge`
- Uses argparse for CLI interface

## Common Issues & Workarounds

### Issue 1: HuggingFace Connection Failures
**Symptom**: "Failed to resolve 'huggingface.co'" or "Max retries exceeded"
**Cause**: No internet access or blocked domain
**Workaround**: Models automatically fall back to template/CPU mode
- Generator: Uses 8 grammatical templates (fast, works offline)
- Discriminator: Uses random predictions
- **Tests still pass with fallback**

### Issue 2: CUDA Out of Memory
**Symptom**: "RuntimeError: CUDA out of memory"
**Solutions**:
1. Use 4-bit quantization: `ComplaintGenerator(use_4bit=True)` (default)
2. Use template mode: `ComplaintGenerator(use_pretrained=False)`
3. Reduce batch size in `config.yaml`: `batch_size: 8`
4. Use CPU: `device="cpu"` in model constructors

### Issue 3: torch-geometric Not Available
**Symptom**: "torch_geometric not available" warnings
**Behavior**: GNN automatically falls back to MLP (Multi-Layer Perceptron)
**Fix (optional)**: `pip install torch-geometric`
**Note**: This is expected - system works without it

### Issue 4: Import Errors After Code Changes
**Symptom**: "ModuleNotFoundError: No module named 'phaita'"
**Solution**: Re-run `pip install -e .` after modifying package structure

### Issue 5: Slow LLM Generation
**Symptom**: Generation takes >30 seconds per complaint
**Cause**: Mistral-7B is inherently slow (autoregressive generation)
**Solution**: Use template mode for faster iteration:
```python
generator = ComplaintGenerator(use_pretrained=False)
```

## Code Modification Guidelines

### When Adding New Conditions
1. Add to `phaita/data/icd_conditions.py` → `RESPIRATORY_CONDITIONS` dict
2. Include: ICD code, name, symptoms, severity indicators, lay terms
3. Update `num_respiratory_conditions` in `config.yaml`
4. Re-run `python test_basic.py` to verify

### When Modifying Models
1. **Generator** (`phaita/models/generator.py`):
   - Inherits from `nn.Module`
   - Must implement: `generate_complaint(symptoms, icd_code)`
   - Supports two modes: LLM (Mistral) and template
2. **Discriminator** (`phaita/models/discriminator.py`):
   - Inherits from `nn.Module`
   - Must implement: `__call__(complaints, return_features=False)`
   - Returns dict with `diagnosis_logits` and `discriminator_scores`

### When Adding Tests
- Use pattern: `def test_name():` returning True/False
- Use `main()` function to run all tests
- Print status with emojis: ✅ (success), ❌ (failure)
- Return exit code: 0 (success), 1 (failure)

### Grammar Rules for Templates
When modifying complaint templates in `generator.py`:
- Use `{main_symptom_gerund}` for "I've been experiencing..."
- Use `{main_symptom_phrase}` for "Really worried about..."
- Use `{main_symptom_action}` for "I can't stop..."
- Use `{other_symptom_phrase}` for secondary symptoms
- See `symptom_grammar_rules` dict for supported symptoms

## Validation Checklist

Before committing changes, ALWAYS run:
1. ✅ `python test_basic.py` - Should complete in ~10 seconds
2. ✅ `python test_enhanced_bayesian.py` - Should complete in ~10 seconds
3. ✅ `python test_forum_scraping.py` - Should complete in ~10 seconds
4. ✅ `python cli.py demo --num-examples 3` - Should complete in ~30 seconds (with fallback)

Optional (if modifying training):
5. `python test_integration.py` - Takes 5+ minutes, may attempt model downloads

## Documentation Files

- **README.md** - Project overview, quick start, architecture
- **IMPLEMENTATION_DETAILS.md** - Deep learning implementation details
- **DEEP_LEARNING_GUIDE.md** - Model architecture, configuration, troubleshooting
- **PROJECT_SUMMARY.md** - Detailed system architecture
- **IMPLEMENTATION_SUMMARY.md** - Testing validation results
- **FIXES_SUMMARY.md** - Critical bug fixes (Tasks 1-4)

## Important Notes

1. **Trust these instructions**: They have been validated by running all commands in a fresh environment
2. **Model downloads are optional**: System works with template fallback mode
3. **Tests are self-contained**: No external test framework required
4. **Installation order matters**: Always run `pip install -r requirements.txt` before `pip install -e .`
5. **Internet access varies**: If HuggingFace is blocked, models fall back gracefully
6. **GPU is optional**: All components work on CPU (template mode is faster anyway)
7. **First run is slower**: Initial runs may compile torch operations

## Quick Reference

| Task | Command | Time | Notes |
|------|---------|------|-------|
| Install deps | `pip install -r requirements.txt` | ~2-3 min | Downloads ~2GB |
| Install package | `pip install -e .` | ~5 sec | Required for imports |
| Run tests | `python test_basic.py` | ~10 sec | Fastest validation |
| Generate data | `python cli.py generate --count 10` | ~10 sec | With fallback |
| Demo | `python cli.py demo --num-examples 3` | ~30 sec | With fallback |
| Simple demo | `python simple_demo.py` | ~5 sec | No downloads |

**When in doubt, run `python test_basic.py` first - it's the fastest way to verify the environment is working.**
