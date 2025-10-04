# Installation Guide

This guide explains the different ways to install PHAITA and its dependencies.

## Overview

PHAITA now supports **modular installation**, allowing you to install only the dependencies you need for your specific use case. This reduces installation size and avoids compatibility issues on systems that don't support certain features (e.g., CPU-only environments).

## Quick Start

### Full Installation (Recommended for Development)
Install everything at once:
```bash
pip install -r requirements.txt
```

### Minimal Installation (CPU-Only, Production)
Install only core dependencies:
```bash
pip install -r requirements-base.txt
```

## Installation Methods

### Method 1: Using requirements files

The most straightforward approach - install from specific requirements files:

```bash
# Install core dependencies only
pip install -r requirements-base.txt

# Then add optional features as needed:
pip install -r requirements-gpu.txt      # Add GPU support
pip install -r requirements-dev.txt      # Add development tools
pip install -r requirements-scraping.txt # Add web scraping
```

### Method 2: Using setup.py with extras

Install using `pip install` with optional extras:

```bash
# Minimal installation (core only)
pip install .
# or for development:
pip install -e .

# Install with GPU support
pip install -e .[gpu]

# Install with development tools
pip install -e .[dev]

# Install with web scraping capabilities
pip install -e .[scraping]

# Install everything
pip install -e .[all]

# Mix and match extras
pip install -e .[gpu,dev]
```

## What's Included in Each Option

### Base Installation (requirements-base.txt)
Core dependencies required for basic PHAITA functionality:
- **torch** (2.5.1) - Deep learning framework
- **transformers** (4.46.0) - HuggingFace models
- **datasets** (≥2.10.0) - Data loading
- **accelerate** (≥0.20.0) - Training acceleration
- **numpy**, **pandas**, **scipy** - Data processing
- **scikit-learn** - Machine learning utilities
- **networkx** - Graph operations
- **matplotlib**, **seaborn** - Visualization
- **tqdm**, **pyyaml**, **safetensors**, **requests** - Utilities

**Size:** ~2-3GB (including PyTorch)

### GPU Extras (requirements-gpu.txt)
GPU-specific features for enhanced performance:
- **bitsandbytes** (0.44.1) - 4-bit model quantization (requires CUDA)
- **torch-geometric** (2.6.1) - Graph Neural Networks

**Requirements:** CUDA-capable GPU with 4GB+ VRAM
**Note:** Without these, the system falls back to:
- Template-based generation (instead of 4-bit quantized models)
- MLP encoder (instead of GNN)

**Additional size:** ~100MB

### Development Extras (requirements-dev.txt)
Tools for testing and development:
- **pytest** (≥7.0) - Testing framework

**Additional size:** ~10MB

### Web Scraping Extras (requirements-scraping.txt)
For forum data collection and web scraping:
- **praw** (≥7.7.0) - Reddit API
- **beautifulsoup4** (≥4.12.0) - HTML parsing

**Additional size:** ~20MB

## Use Cases

### I want to... run PHAITA on a CPU-only server
```bash
pip install -r requirements-base.txt
pip install -e .
```
The system will automatically use template-based fallbacks and MLP encoders.

### I want to... develop and run tests locally
```bash
pip install -e .[dev]
# or
pip install -r requirements-base.txt
pip install -r requirements-dev.txt
```

### I want to... train models with GPU acceleration
```bash
pip install -e .[gpu,dev]
# or
pip install -r requirements-base.txt
pip install -r requirements-gpu.txt
pip install -r requirements-dev.txt
```

### I want to... scrape forum data for training
```bash
pip install -e .[scraping,dev]
# or
pip install -r requirements-base.txt
pip install -r requirements-scraping.txt
pip install -r requirements-dev.txt
```

### I want to... do everything (development, GPU, scraping)
```bash
pip install -r requirements.txt
# or
pip install -e .[all]
```

## Compatibility Notes

### CPU-Only Environments
- ✅ Base installation works on any system with Python 3.10+
- ⚠️ Skip `requirements-gpu.txt` or the `[gpu]` extra
- ⚠️ `bitsandbytes` installation may succeed but won't function without CUDA

### GPU Environments
- ✅ Install CUDA 11.8+ before installing GPU extras
- ✅ Verify: `python -c "import torch; print(torch.cuda.is_available())"`
- ⚠️ `bitsandbytes` requires a CUDA-compatible GPU

### Windows
- ✅ Most dependencies work on Windows
- ⚠️ `bitsandbytes` has limited Windows support (prefer WSL2)

### macOS
- ✅ Base installation works on macOS
- ⚠️ CUDA not available - skip GPU extras

## Verifying Installation

After installation, verify everything works:

```bash
# Test core functionality
python tests/test_basic.py

# Test modular dependencies configuration
python tests/test_modular_dependencies.py

# Test installation and imports
python tests/test_installation_integration.py

# Quick Python check
python -c "from phaita.data.icd_conditions import RespiratoryConditions; print('✅ PHAITA installed correctly!')"
```

## Troubleshooting

### "No module named 'bitsandbytes'"
- If you need GPU features: `pip install -r requirements-gpu.txt`
- If CPU-only: Ignore this - the system will use fallbacks

### "torch_geometric not found"
- If you need GNN features: `pip install torch-geometric==2.6.1`
- If CPU-only: Ignore this - the system will use MLP fallback

### "CUDA not available"
- Verify GPU: `nvidia-smi`
- Verify PyTorch sees GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- If no GPU, use CPU-only installation (skip `requirements-gpu.txt`)

### Installation takes too long
- Use PyTorch index for faster downloads:
  ```bash
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements-base.txt
  ```

## Migration from Old Installation

If you previously installed using the old `requirements.txt`, no action needed! The file still includes all dependencies for backward compatibility.

To switch to minimal installation:
```bash
pip uninstall bitsandbytes torch-geometric pytest praw beautifulsoup4
pip install -r requirements-base.txt
```

## See Also

- [README.md](../README.md) - Main documentation
- [docs/guides/SOP.md](docs/guides/SOP.md) - Standard Operating Procedure
- [DEEP_LEARNING_GUIDE.md](../DEEP_LEARNING_GUIDE.md) - GPU setup and troubleshooting
