# PHAITA Testing Guide

Comprehensive guide to all test suites in the PHAITA project.

## Quick Start

```bash
# Install dependencies first
pip install -r requirements.txt

# Run core test suites (recommended order)
python test_basic.py                      # ~10s - Core data, config, Bayesian
python test_enhanced_bayesian.py          # ~10s - Age/severity/rare presentations
python test_forum_scraping.py             # ~10s - Forum data augmentation
python test_dialogue_engine.py            # ~5s  - Belief updating, information gain
python test_diagnosis_orchestrator.py     # ~3s  - Ensemble, red-flags, escalation
python test_conversation_flow.py          # ~5s  - End-to-end triage sessions
python test_escalation_guidance.py        # ~3s  - Care routing validation
```

**Note:** All tests are plain Python scripts - **NO pytest required**.

## Test Categories

### Core Tests (Always Run These)

#### `test_basic.py` (~10 seconds)
**Purpose:** Validates core data layer, configuration, and Bayesian network logic.

**What it tests:**
- Medical conditions data layer (10 ICD-10 respiratory conditions)
- Configuration system (YAML loading, model configs)
- Bayesian symptom network initialization
- Synthetic data generation basics

**No dependencies:** Does NOT require transformer models or GPU.

**Expected results:** 4/4 tests pass

---

#### `test_enhanced_bayesian.py` (~10 seconds)
**Purpose:** Tests advanced Bayesian network features.

**What it tests:**
- Age-based symptom probability adjustments (pediatric, geriatric)
- Severity modulation (mild, moderate, severe)
- Rare presentation generation
- Comorbidity modeling (diabetes, hypertension, etc.)
- Priority-based symptom trimming

**No heavy dependencies:** CPU-only, no model downloads.

**Expected results:** 6/6 tests pass

---

#### `test_forum_scraping.py` (~10 seconds)
**Purpose:** Validates forum data augmentation and lay language mapping.

**What it tests:**
- Bidirectional medical ↔ lay term mapping
- Forum-style text generation
- Data augmentation for training corpus
- Vocabulary expansion

**No dependencies:** Pure data layer testing.

**Expected results:** 3/3 tests pass

---

### Dialogue and Conversation Tests

#### `test_dialogue_engine.py` (~5 seconds)
**Purpose:** Unit tests for multi-turn dialogue components.

**What it tests:**
- Belief updating with symptom evidence
- Information gain calculation for question selection
- Bayesian probability updates
- Symptom tracking (confirmed/denied)
- State management across turns

**Dependencies:** BayesianSymptomNetwork only (no transformers).

**Expected results:** All unit tests pass

---

#### `test_conversation_flow.py` (~5 seconds)
**Purpose:** End-to-end integration tests for complete triage sessions.

**What it tests:**
- Complete triage workflow (symptom → questions → diagnosis)
- Multi-turn dialogue progression
- Early termination on confidence threshold
- Maximum turn limit enforcement
- Edge cases (deny all symptoms, conflicting symptoms)
- Differential diagnosis generation

**Integration test:** Uses actual DialogueEngine implementation.

**Expected results:** 5/5 tests pass

**Detailed coverage:**
1. ✅ Complete asthma triage session
2. ✅ Edge case: deny all symptoms
3. ✅ Edge case: conflicting symptom evidence
4. ✅ Early termination when confidence > threshold
5. ✅ Maximum turn limit prevents infinite loops

See [TESTING_MULTI_TURN_DIALOGUES.md](TESTING_MULTI_TURN_DIALOGUES.md) for detailed documentation.

---

### Diagnosis and Orchestration Tests

#### `test_diagnosis_orchestrator.py` (~3 seconds)
**Purpose:** Tests the diagnosis orchestrator with red-flag integration.

**What it tests:**
- Ensemble diagnosis (Bayesian + neural predictions)
- Red-flag symptom detection
- Escalation level determination (emergency/urgent/routine)
- Condition-specific red-flags (10 respiratory conditions)
- Guidance text generation
- Care routing logic

**Dependencies:** Uses stub models, no transformers required.

**Expected results:** 11/11 tests pass

**Key features tested:**
- ✅ Red-flag detection and matching
- ✅ Multi-condition red-flag support
- ✅ Escalation level determination
- ✅ Emergency/urgent/routine routing
- ✅ Guidance text generation
- ✅ Probability-based escalation

---

#### `test_escalation_guidance.py` (~3 seconds)
**Purpose:** Integration tests for red-flag detection and care routing.

**What it tests:**
- Emergency red-flags trigger appropriate escalation
- Urgent care routing for moderate probability cases
- Routine care guidance for low probability
- Condition-specific actions in guidance text
- Multiple escalation pathways (red-flags, probability, condition type)

**Expected results:** 6/6 tests pass

**Test scenarios:**
1. 🚑 Emergency red-flags (severe respiratory distress, unable to speak)
2. ⚠️ Urgent care (moderate probability, no red-flags)
3. 📋 Routine care (low probability, monitoring advice)
4. 🏥 Condition-specific guidance validation
5. 🚑 Pneumonia-specific red-flags (confusion, low O2)
6. 🚑 Emergency conditions with high probability

---

### Model and Architecture Tests

#### `test_causal_graph.py` (~5 seconds)
**Purpose:** Tests causal graph functionality in GNN module.

**What it tests:**
- Causal edge loading from config
- Temporal edge relationships
- Edge type embeddings
- Graph construction with causality
- Message passing with edge types

**Dependencies:** Requires PyTorch and torch-geometric.

**Expected results:** 6/6 tests pass

---

#### `test_gnn_performance.py` (~10 seconds)
**Purpose:** Benchmarks and validates GNN module performance.

**What it tests:**
- Graph construction performance
- Forward pass speed
- Memory efficiency
- Edge type handling
- Attention mechanism functionality

**Dependencies:** Requires PyTorch and torch-geometric.

**GPU optional:** Works on CPU but faster with GPU.

---

#### `test_template_diversity.py` (~5 seconds)
**Purpose:** Tests template-based complaint generation system.

**What it tests:**
- Template diversity (28 patterns)
- Intelligent template selection (age, severity, formality)
- Uniqueness in generated complaints (>80%)
- Grammar correctness
- Template category coverage
- Recent template tracking (avoids last 5)

**No dependencies:** CPU-only, no models required.

**Expected results:** 6/6 tests pass

**Performance targets:**
- Total templates: 28
- Uniqueness (1000 gen): >81%
- Max template usage: <6%
- Grammar errors: 0

---

#### `test_temporal_modeling.py` (~10 seconds)
**Purpose:** Tests temporal symptom progression modeling.

**What it tests:**
- SymptomTimeline tracking
- LSTM-based temporal encoder
- Temporal pattern matching for conditions
- DialogueEngine integration with temporal module
- Diagnosis accuracy improvement with temporal data

**Dependencies:** Requires PyTorch (LSTM module).

**Expected results:** 5/5 tests pass

---

#### `test_uncertainty.py` (~5 seconds)
**Purpose:** Tests uncertainty quantification in predictions.

**What it tests:**
- Monte Carlo dropout for uncertainty estimation
- Confidence intervals
- Epistemic vs aleatoric uncertainty
- Prediction reliability scoring
- Calibration metrics

**Dependencies:** Requires PyTorch.

---

### CLI and Workflow Tests

#### `test_cli_challenge_command.py` (~2 seconds)
**Purpose:** Tests the challenge command-line interface.

**What it tests:**
- Challenge mode parameter handling
- Age group and severity flags
- Rare presentation generation
- Command parsing and validation

**No heavy dependencies:** Uses stub models.

---

#### `test_cli_triage_workflow.py` (~5 seconds)
**Purpose:** Tests the diagnose command interactive workflow.

**What it tests:**
- Information gain loop
- Interactive question-answer flow
- Diagnosis prediction integration
- Command-line interface behavior

**Dependencies:** Uses mocks/stubs.

---

#### `test_patient_cli.py` (~2 seconds)
**Purpose:** Tests the patient-facing CLI interface.

**What it tests:**
- Patient simulation workflow
- User interaction patterns
- Command parsing

---

#### `test_patient_simulation.py` (~2 seconds)
**Purpose:** Tests patient simulation module.

**What it tests:**
- Virtual patient generation
- Symptom response simulation
- Realistic patient behavior

---

### Integration Tests

#### `test_integration.py` (~2 minutes, requires network)
**Purpose:** Full-stack integration test with real models.

**What it tests:**
- Model downloading from HuggingFace Hub
- Mistral 7B complaint generation
- DeBERTa diagnosis prediction
- End-to-end adversarial pipeline
- GPU/CPU compatibility

**⚠️ Warning:** Attempts model downloads, retries ~60s, then falls back gracefully.

**Network required:** Downloads ~7GB of models on first run.

**Expected behavior:** May timeout on Task 1 (model download), fallback is expected.

---

#### `test_conversation_engine.py` (~5 seconds)
**Purpose:** Regression tests for conversation engine triage flow.

**What it tests:**
- Conversation engine logic (stub-based)
- Triage flow orchestration
- Turn management
- State transitions

**No transformers required:** Uses stubs for model components.

---

#### `test_triage_differential.py` (~2 seconds)
**Purpose:** Tests differential diagnosis generation.

**What it tests:**
- Top-k diagnosis ranking
- Probability normalization
- Condition filtering

---

### Configuration and Setup Tests

#### `test_conditions_config.py` (~5 seconds)
**Purpose:** Tests configurable respiratory conditions system.

**What it tests:**
- YAML config loading
- Custom config paths (PHAITA_RESPIRATORY_CONFIG env var)
- Hot-reload functionality
- Config validation
- Integration with SymptomGenerator and ForumDataAugmentation

**Dependencies:** Requires pytest (exception: uses pytest framework).

---

#### `test_model_loader.py` (~10 seconds)
**Purpose:** Tests the model loading and initialization system.

**What it tests:**
- Model loader utilities
- Quantization support
- Device management (CPU/GPU)
- Model caching
- Error handling for missing models

**Dependencies:** Requires transformers library.

---

### Training and Adversarial Tests

#### `test_adversarial_trainer_dataset.py` (~10 seconds)
**Purpose:** Tests adversarial training dataset and trainer integration.

**What it tests:**
- Dataset construction for adversarial training
- Generator/discriminator integration
- Training loop functionality
- Gradient flow
- Loss computation

**Dependencies:** Requires PyTorch.

---

## Test Execution Strategies

### Quick Validation (recommended for CI)
```bash
# Core functionality - no model downloads required (~30 seconds total)
python test_basic.py
python test_enhanced_bayesian.py
python test_forum_scraping.py
python test_dialogue_engine.py
python test_diagnosis_orchestrator.py
```

### Full Test Suite (local development)
```bash
# Run all tests except integration (~2 minutes)
for test in test_*.py; do
    if [ "$test" != "test_integration.py" ]; then
        echo "Running $test..."
        python "$test" || echo "FAILED: $test"
    fi
done
```

### Integration Test (requires network, ~2-5 minutes)
```bash
# Only run when testing full stack with model downloads
python test_integration.py
```

### Feature-Specific Testing
```bash
# Testing dialogue features
python test_dialogue_engine.py
python test_conversation_flow.py
python test_escalation_guidance.py
python test_conversation_engine.py

# Testing Bayesian features
python test_basic.py          # Includes Bayesian tests
python test_enhanced_bayesian.py

# Testing GNN features
python test_causal_graph.py
python test_gnn_performance.py

# Testing template system
python test_template_diversity.py

# Testing temporal features
python test_temporal_modeling.py
```

## Test Dependencies

### No External Dependencies
These tests work without any model downloads or heavy libraries:
- `test_basic.py` (pure Python, no torch imports in test code)
- `test_forum_scraping.py` (data layer only)
- `test_cli_challenge_command.py` (stubs)
- `test_cli_triage_workflow.py` (mocks)

### Requires PyTorch Only
- `test_enhanced_bayesian.py` (may import torch modules)
- `test_causal_graph.py`
- `test_adversarial_trainer_dataset.py`
- `test_temporal_modeling.py`
- `test_uncertainty.py`

### Requires Transformers Library
- `test_model_loader.py`

### Requires torch-geometric
- `test_causal_graph.py`
- `test_gnn_performance.py`

### Requires Network Access
- `test_integration.py` (downloads models from HuggingFace Hub)

## Common Issues and Troubleshooting

### Issue: ImportError: No module named 'torch'
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: test_integration.py times out
**Expected behavior:** Model download can take 2-5 minutes on first run. Subsequent runs use cached models.

**If it fails:** This is expected offline or with network issues. The system falls back gracefully.

### Issue: CUDA out of memory
**Solution:** Models are 4-bit quantized by default. If still OOM:
- Close other GPU applications
- Reduce batch size in config.yaml
- Use CPU-only mode: `use_4bit=False`

### Issue: Tests fail with "No module named 'pytest'"
**Solution:** Only `test_conditions_config.py` uses pytest. All other tests are plain Python:
```bash
pip install pytest  # Only needed for test_conditions_config.py
```

### Issue: Red-flags not detected in tests
**Cause:** Symptom normalization mismatch (underscores vs spaces).

**Solution:** The code uses `_normalize_symptom()` method for comparison. This is already handled in current tests.

## Test Output Format

All tests follow PHAITA conventions with emoji indicators:

```
🏥 PHAITA Test Suite
============================================================
🧪 Testing Data Layer...
   ✓ Found 10 respiratory conditions
   ✓ Validated ICD-10 codes
   ✓ Checked symptom vocabulary
✅ Data layer tests passed

🧪 Testing Bayesian Network...
   ✓ Network initialized successfully
   ✓ Symptom probabilities computed
✅ Bayesian network tests passed

============================================================
📊 Test Results: 4/4 tests passed
🎉 All tests passed!
```

## Contributing New Tests

When adding new tests to PHAITA:

1. **Naming:** Use `test_<feature>.py` convention
2. **Shebang:** Start with `#!/usr/bin/env python3`
3. **Docstring:** Include module docstring explaining purpose
4. **Plain Python:** No pytest/unittest - use plain functions
5. **Output:** Use emoji indicators (🧪, ✅, ❌)
6. **Assertions:** Use clear assertion messages
7. **Print debugging:** Include intermediate values for debugging
8. **Self-contained:** Minimize external dependencies
9. **Fast:** Keep tests < 10 seconds if possible
10. **Document:** Add to this guide with timing and purpose

### Example Test Structure

```python
#!/usr/bin/env python3
"""
Test script for <feature>.
<Brief description of what this test validates>
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_feature_one():
    """Test <specific aspect>."""
    print("🧪 Testing <aspect>...")
    
    # Test logic here
    assert condition, "Clear failure message"
    
    print("  ✓ <what passed>")
    return True

def main():
    """Run all tests."""
    print("🏥 PHAITA <Feature> Test Suite")
    print("=" * 60)
    
    tests = [
        ("Feature One", test_feature_one),
        ("Feature Two", test_feature_two),
    ]
    
    passed = 0
    for name, test_fn in tests:
        try:
            if test_fn():
                print(f"✅ {name}")
                passed += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

## Related Documentation

- [TESTING_MULTI_TURN_DIALOGUES.md](TESTING_MULTI_TURN_DIALOGUES.md) - Detailed dialogue test documentation
- [architecture/DIALOGUE_ENGINE.md](architecture/DIALOGUE_ENGINE.md) - DialogueEngine architecture
- [architecture/DIAGNOSIS_ORCHESTRATOR_README.md](architecture/DIAGNOSIS_ORCHESTRATOR_README.md) - Red-flag system
- [../IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) - Module overview

## Summary

PHAITA includes **22 comprehensive test files** covering:
- ✅ Core data layer and configuration
- ✅ Bayesian networks (basic + enhanced features)
- ✅ Multi-turn dialogue systems
- ✅ Diagnosis orchestration with red-flags
- ✅ Graph neural networks with causal edges
- ✅ Template generation systems
- ✅ Temporal symptom modeling
- ✅ Uncertainty quantification
- ✅ CLI interfaces and workflows
- ✅ Full integration tests

**Total test coverage:** ~22 test files, 100+ individual test cases, ~5 minutes for full suite (excluding integration test).

All tests are plain Python scripts with clear output and minimal dependencies. No pytest required (except `test_conditions_config.py`).
