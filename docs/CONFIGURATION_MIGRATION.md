# Configuration Migration Guide

**Date**: January 2025  
**Version**: PHAITA v0.1.0+

## Overview

PHAITA's configuration structure has been refactored for better organization and physician usability. This guide helps you migrate from the old structure to the new one.

## What Changed

### Old Structure (Pre-Migration)
```
config/
├── respiratory_conditions.yaml
├── red_flags.yaml
├── comorbidity_effects.yaml
├── symptom_causality.yaml
└── temporal_patterns.yaml

phaita/data/
└── templates.yaml

config.yaml  # System configuration
```

### New Structure (Current)
```
config/
├── system.yaml                  # NEW: Technical configuration
├── medical_knowledge.yaml       # NEW: Consolidated medical knowledge
├── templates.yaml               # MOVED: From phaita/data/
├── respiratory_conditions.yaml  # LEGACY: Still supported
├── red_flags.yaml              # LEGACY: Still supported
├── comorbidity_effects.yaml    # LEGACY: Still supported
├── symptom_causality.yaml      # LEGACY: Still supported
└── temporal_patterns.yaml      # LEGACY: Still supported

config.yaml  # LEGACY: Still supported
```

## Key Changes

### 1. System Configuration

**Old**: `config.yaml` (mixed system and module settings)

**New**: `config/system.yaml` (dedicated system configuration)

```yaml
# config/system.yaml
model:
  deberta_model: "microsoft/deberta-base"
  mistral_model: "mistralai/Mistral-7B-Instruct-v0.2"
  gnn_hidden_dim: 128
  gnn_num_layers: 3
  use_quantization: true

training:
  num_epochs: 100
  batch_size: 16
  generator_lr: 2.0e-5
  discriminator_lr: 1.0e-4
  diversity_weight: 0.1
  eval_interval: 10
  save_interval: 50
  device: null  # auto-detect

conversation:  # NEW SECTION
  max_questions: 10
  confidence_threshold: 0.85
  min_info_gain: 0.1
  enable_red_flag_escalation: true

triage:  # NEW SECTION
  max_diagnoses: 10
  min_confidence: 0.05
  enable_red_flag_check: true
  enable_info_sheets: true
  escalation_thresholds:
    critical: 0.95
    urgent: 0.80
    routine: 0.50
```

### 2. Medical Knowledge Configuration

**Old**: Five separate files in `config/`

**New**: One consolidated file `config/medical_knowledge.yaml`

```yaml
# config/medical_knowledge.yaml
conditions:
  J45.9:
    name: Asthma
    symptoms:
      - wheezing
      - shortness_of_breath
    # ... rest of condition data

red_flags:
  J45.9:
    red_flags:
      - severe_respiratory_distress
    # ... rest of red-flag data

comorbidity_effects:
  diabetes:
    symptom_modifiers:
      fatigue: 1.3
    # ... rest of comorbidity data

symptom_causality:
  causal_edges:
    - source: airway_inflammation
      target: wheezing
    # ... rest of causality data

temporal_patterns:
  J45.9:
    typical_progression:
      - symptom: wheezing
        onset_hour: 0
    # ... rest of temporal data
```

### 3. Template Configuration

**Old**: `phaita/data/templates.yaml`

**New**: `config/templates.yaml`

Templates file moved to config directory for consistency.

## Migration Steps

### Option 1: Use New Structure (Recommended)

No migration needed! The system automatically uses:
- `config/medical_knowledge.yaml` if it exists
- Falls back to individual files if not

**To adopt the new structure:**

1. Files are already created during installation
2. Edit `config/medical_knowledge.yaml` for medical knowledge
3. Edit `config/system.yaml` for system settings
4. Templates automatically loaded from `config/templates.yaml`

### Option 2: Continue Using Legacy Files

No changes required! The system maintains full backward compatibility:
- Individual config files still work
- Legacy `config.yaml` still supported
- No breaking changes

**When to use legacy mode:**
- You have existing custom configurations
- You prefer separate files for different concerns
- You're using environment variables like `PHAITA_RESPIRATORY_CONFIG`

## Code Changes

### Configuration Loading

**Old**:
```python
from phaita.utils import Config

# Required explicit path
config = Config.from_yaml("config.yaml")
```

**New**:
```python
from phaita.utils import Config

# Auto-detects system.yaml or config.yaml
config = Config.from_yaml()  # Optional: specify path if needed
```

### Medical Configuration Access

No changes needed! All existing code continues to work:

```python
from phaita.data import RespiratoryConditions

# Still works exactly the same
conditions = RespiratoryConditions.get_all_conditions()

# Hot-reload still works
RespiratoryConditions.reload()
```

### Template Loading

**Old**:
```python
from phaita.data.template_loader import TemplateManager

# Explicitly pointed to phaita/data/
manager = TemplateManager()
```

**New**:
```python
from phaita.data.template_loader import TemplateManager

# Auto-detects config/templates.yaml or phaita/data/templates.yaml
manager = TemplateManager()  # No changes needed!
```

## Testing Your Migration

Run the test suite to verify everything works:

```bash
# Core functionality
python tests/test_basic.py

# Enhanced features
python tests/test_enhanced_bayesian.py

# Configuration-specific
python tests/test_conditions_config.py
```

All tests should pass with either configuration structure.

## Benefits of New Structure

### For Physicians
- ✅ **One file to edit**: All medical knowledge in `config/medical_knowledge.yaml`
- ✅ **Clear sections**: Conditions, red-flags, comorbidity, causality, temporal all in one place
- ✅ **Better documentation**: Header comments explain each section
- ✅ **Easier hot-reload**: Reload once affects all medical knowledge

### For Developers
- ✅ **Clearer separation**: System config vs. medical config
- ✅ **Easier version control**: Related changes grouped together
- ✅ **Better organization**: Templates with other configs
- ✅ **Backward compatible**: Legacy code still works

### For Deployment
- ✅ **Fewer files to manage**: 3 files instead of 6+
- ✅ **Easier updates**: Update medical knowledge in one place
- ✅ **Configuration as code**: More maintainable
- ✅ **Environment flexibility**: Can still use individual files if needed

## Troubleshooting

### "Config file not found"
- System looks for `config/system.yaml` first, then `config.yaml`
- Place your config in either location

### "Medical knowledge not loading"
- System looks for `config/medical_knowledge.yaml` first
- Falls back to individual files in `config/`
- Check file exists and has correct YAML syntax

### "Templates not found"
- System looks for `config/templates.yaml` first
- Falls back to `phaita/data/templates.yaml`
- Ensure file is in one of these locations

### Environment Variable Override
```bash
# Override respiratory conditions path
export PHAITA_RESPIRATORY_CONFIG=/path/to/custom/conditions.yaml

# System respects this for medical_knowledge.yaml too
```

## Frequently Asked Questions

**Q: Do I need to migrate immediately?**  
A: No, the old structure still works. Migrate when convenient.

**Q: Can I mix old and new structures?**  
A: Yes, but not recommended. System will use new structure if available, fall back to old.

**Q: Will this break my existing code?**  
A: No, all APIs remain the same. Only file locations changed.

**Q: How do I edit medical knowledge now?**  
A: Edit `config/medical_knowledge.yaml` (preferred) or continue using individual files.

**Q: What about my custom configurations?**  
A: They continue to work. Use `PHAITA_RESPIRATORY_CONFIG` to point to custom files.

## Support

If you encounter issues:
1. Check this guide
2. See [docs/DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
3. Run tests: `python tests/test_basic.py`
4. Check [docs/guides/SOP.md](guides/SOP.md) for troubleshooting

---

**Last Updated**: January 2025  
**Applies To**: PHAITA v0.1.0 and later
