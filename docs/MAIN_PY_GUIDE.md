# main.py Entry Point Guide

## Overview

PHAITA now includes a centralized entry point (`main.py`) that simplifies access to the most common tasks. This provides an easier way to get started while maintaining backward compatibility with the existing `cli.py` and `patient_cli.py` interfaces.

## Quick Start

```bash
# See all available commands
python main.py --help

# Run a demo (best way to get started)
python main.py demo

# Get help on any command
python main.py <command> --help
```

## Available Commands

### `demo` - Run Simple Demonstration
Runs the simple demo script without any dependencies required.

```bash
python main.py demo
```

**What it does:** Shows respiratory conditions, symptom generation, and basic data layer functionality.

### `train` - Train the Model
Train the adversarial model with custom hyperparameters.

```bash
python main.py train --epochs 50 --batch-size 16 --lr 0.001
```

**Options:**
- `--epochs` - Number of training epochs (default: from config)
- `--batch-size` - Training batch size (default: from config)
- `--lr` - Learning rate (default: from config)
- `-v, --verbose` - Enable verbose logging

### `diagnose` - Run Diagnosis/Triage
Diagnose patient complaints using the trained model.

```bash
# Single complaint
python main.py diagnose --complaint "I can't breathe"

# Interactive mode (recommended)
python main.py diagnose --interactive

# With detailed analysis
python main.py diagnose --complaint "chest pain" --detailed
```

**Options:**
- `--complaint` - Patient complaint text to analyze
- `--interactive` - Interactive mode for multiple complaints
- `--detailed` - Show detailed analysis
- `-v, --verbose` - Enable verbose logging

### `interactive` - Patient Simulation
Run an interactive patient simulation where you play the clinician.

```bash
# Random patient
python main.py interactive

# Specific condition
python main.py interactive --condition J45.9

# With custom settings
python main.py interactive --condition J18.9 --seed 42 --max-turns 10
```

**Options:**
- `--condition` - ICD-10 condition code to simulate (e.g., J45.9 for Asthma)
- `--seed` - Random seed for reproducibility
- `--max-turns` - Maximum number of conversation turns

### `generate` - Generate Synthetic Data
Generate synthetic patient data for training or testing.

```bash
# Generate 10 examples
python main.py generate --count 10

# Generate for specific condition
python main.py generate --condition J45.9 --count 5

# Save to file
python main.py generate --count 100 --output data.json
```

**Options:**
- `--count` - Number of examples to generate (default: 10)
- `--condition` - Specific condition to generate for (optional)
- `--output` - Output file path (optional)
- `-v, --verbose` - Enable verbose logging

### `cli` - Access Advanced Features
Forward to the full CLI for advanced features not available in the simplified interface.

```bash
# Show full CLI help
python main.py cli --help

# Use advanced features
python main.py cli challenge --rare-cases 5 --show-failures
python main.py cli conversation --symptoms "cough,fever"
```

## Architecture

The `main.py` entry point uses a **delegation model** - it doesn't duplicate any functionality, it simply provides a cleaner interface to existing tools:

```
main.py (simplified interface)
├── demo → runs demos/simple_demo.py
├── train → delegates to cli.py train
├── diagnose → delegates to cli.py diagnose
├── interactive → delegates to patient_cli.py
├── generate → delegates to cli.py generate
└── cli → forwards to cli.py with arguments
```

This design ensures:
- **No code duplication** - all logic remains in original files
- **Backward compatibility** - cli.py and patient_cli.py work exactly as before
- **Easy maintenance** - changes to cli.py automatically benefit main.py
- **Minimal footprint** - main.py is ~300 lines, mostly argument parsing

## When to Use Each Interface

### Use `main.py` when:
- ✅ You're new to PHAITA
- ✅ You want quick access to common tasks
- ✅ You prefer a simplified interface
- ✅ You're running demos or quick tests

### Use `cli.py` when:
- ✅ You need advanced features (challenge mode, conversation mode, etc.)
- ✅ You want fine-grained control over options
- ✅ You're doing development work
- ✅ You need features not exposed in main.py

### Use `patient_cli.py` when:
- ✅ You need direct access to patient simulation with all options
- ✅ You want specific simulation strategies (brief, detailed, etc.)
- ✅ You're testing the patient agent specifically

## Examples

### First Time User Journey

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run demo to see what PHAITA does
python main.py demo

# 3. Try interactive diagnosis
python main.py diagnose --interactive
# Enter: "I have a terrible cough and fever"
# Follow the prompts...

# 4. Try patient simulation
python main.py interactive
# You're now the clinician interviewing a simulated patient

# 5. Ready for more? Use the full CLI
python main.py cli --help
```

### Development Workflow

```bash
# Generate training data
python main.py generate --count 100 --output training_data.json

# Train model
python main.py train --epochs 50 --batch-size 16

# Test on specific cases
python main.py diagnose --complaint "shortness of breath"
python main.py diagnose --complaint "chest tightness"

# Use advanced features for validation
python cli.py challenge --rare-cases 5 --atypical-cases 5 --verbose
```

## Testing

The main.py entry point is thoroughly tested in `tests/test_main_entry_point.py`:

```bash
# Run tests
python tests/test_main_entry_point.py
```

Tests verify:
- All commands show correct help
- Arguments are properly forwarded
- Demo execution works
- All subcommands are accessible

## Implementation Notes

- **Language:** Python 3.10+
- **Dependencies:** None beyond PHAITA's existing dependencies
- **Size:** ~300 lines of code
- **Tests:** 9 comprehensive tests, all passing
- **Documentation:** README.md, docs/guides/SOP.md updated

## Future Enhancements

Possible future additions to main.py:
- `web` command - Start web interface
- `test` command - Run test suite
- `config` command - Interactive configuration
- `status` command - Check system status

These would be added based on user feedback and usage patterns.
