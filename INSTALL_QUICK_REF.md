# PHAITA Installation Quick Reference

## One-Line Installs

```bash
# Full installation (everything)
pip install -r requirements.txt

# Minimal installation (CPU-only, core features)
pip install -r requirements-base.txt

# With setup.py extras
pip install -e .[all]      # Everything
pip install -e .[gpu]      # Core + GPU
pip install -e .[dev]      # Core + Testing
pip install -e .[scraping] # Core + Web scraping
pip install -e .[gpu,dev]  # Mix and match
```

## What You Get

| Installation | Size | CUDA? | Features |
|--------------|------|-------|----------|
| `requirements-base.txt` | ~2-3GB | No | Core ML, CPU operation, template fallbacks |
| + `requirements-gpu.txt` | +100MB | **Yes** | 4-bit quantization, GNN module |
| + `requirements-dev.txt` | +10MB | No | pytest testing framework |
| + `requirements-scraping.txt` | +20MB | No | Reddit API, web scraping |
| `requirements.txt` (all) | ~3-4GB | Optional | Everything (backward compatible) |

## Quick Decision Tree

**Do you have a CUDA GPU?**
- ✅ Yes → `pip install -e .[gpu,dev]` or `requirements.txt`
- ❌ No → `pip install -r requirements-base.txt` or `pip install -e .`

**Need to run tests?**
- Add `[dev]` extra or `requirements-dev.txt`

**Need forum scraping?**
- Add `[scraping]` extra or `requirements-scraping.txt`

## Verify Installation

```bash
# Quick check
python -c "from phaita.data.icd_conditions import RespiratoryConditions; print('✅ Works!')"

# Full test suite
python tests/test_basic.py
python tests/test_modular_dependencies.py
python tests/test_installation_integration.py
```

## Common Issues

| Error | Solution |
|-------|----------|
| "No module named 'bitsandbytes'" | Either install GPU extras or ignore (uses CPU fallback) |
| "torch_geometric not found" | Either install GPU extras or ignore (uses MLP fallback) |
| "CUDA not available" | Use CPU-only install: `requirements-base.txt` |
| Tests fail with import errors | Run `pip install -e .` to install package in editable mode |

## See Full Guide

📖 [INSTALLATION.md](INSTALLATION.md) - Complete installation guide with troubleshooting
