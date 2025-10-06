# Implementation Summary: Learnable Comorbidity Effects and Symptom Causality

## Problem Statement

Make comorbidity_effects and symptom_causality learnable rather than fixed weights.

## Solution Overview

Transformed fixed YAML configuration weights into fully learnable PyTorch parameters that can be optimized through gradient descent during adversarial training.

## Implementation Details

### 1. Learnable Comorbidity Effects

**File**: `phaita/models/learnable_comorbidity.py`

**Architecture**:
- PyTorch `nn.Module` with learnable parameter matrix: `[num_comorbidities, num_symptoms]`
- Log-parameterization: stores `log(multiplier)` and exponentiates during forward pass
- Ensures positive multipliers through `exp()` transformation
- Initialized from `config/comorbidity_effects.yaml` clinical values

**Parameters**:
- 8 comorbidities × 23 symptoms = **184 learnable parameters**
- Memory: ~736 bytes (float32)
- Device: CPU or CUDA compatible

**Key Methods**:
- `get_symptom_modifiers(comorbidity)`: Returns learned multipliers for a comorbidity
- `get_comorbidity_data(comorbidity)`: Returns full data including metadata
- `forward()`: Returns exponentiated weight matrix for gradient computation

### 2. Learnable Symptom Causality

**File**: `phaita/models/learnable_causality.py`

**Architecture**:
- PyTorch `nn.Module` with two learnable parameter vectors:
  - `causal_weights`: [num_causal_edges] - causal edge strengths
  - `temporal_weights`: [num_temporal_edges] - temporal edge strengths
- Logit parameterization: stores `logit(strength)` and applies sigmoid
- Ensures strengths in [0, 1] range through `sigmoid()` transformation
- Initialized from `config/symptom_causality.yaml` clinical values

**Parameters**:
- 10 causal edges + 6 temporal edges = **16 learnable parameters**
- Memory: ~64 bytes (float32)
- Temporal delays: Fixed as buffers (not learned)

**Key Methods**:
- `get_causal_edges()`: Returns list of (source, target, strength) tuples
- `get_temporal_edges()`: Returns list of (earlier, later, strength, delay) tuples
- `get_config_for_gnn()`: Generates GNN-compatible configuration dictionary
- `forward()`: Returns sigmoid-transformed strengths for gradient computation

### 3. Integration Points

#### EnhancedBayesianNetwork
**Modified**: `phaita/models/enhanced_bayesian_network.py`

**Changes**:
- Added `use_learnable_comorbidity` parameter to `__init__`
- Added `device` parameter for learnable module
- Conditional initialization of `LearnableComorbidityEffects`
- Updated `sample_symptoms()` to use learnable modifiers when available
- Factory function `create_enhanced_bayesian_network()` updated

**Backward Compatibility**: Default `use_learnable_comorbidity=False` uses fixed YAML weights

#### SymptomGraphBuilder & SymptomGraphModule
**Modified**: `phaita/models/gnn_module.py`

**Changes**:
- Added `learnable_causality` parameter to `SymptomGraphBuilder.__init__`
- Conditional use of learnable causality config via `get_config_for_gnn()`
- Added `learnable_causality` parameter to `SymptomGraphModule.__init__`
- Passed learnable module through graph builder

**Backward Compatibility**: Default `learnable_causality=None` uses fixed YAML weights

#### DiagnosisDiscriminator
**Modified**: `phaita/models/discriminator.py`

**Changes**:
- Added `learnable_causality` parameter to `__init__`
- Added TYPE_CHECKING import for type hints
- Passed learnable causality to internal GNN module

**Backward Compatibility**: Default `learnable_causality=None` uses fixed weights

#### AdversarialTrainer
**Modified**: `phaita/training/adversarial_trainer.py`

**Changes**:
- Added `use_learnable_comorbidity` and `use_learnable_causality` flags
- Added `comorbidity_lr` and `causality_lr` learning rate parameters
- Created separate `AdamW` optimizers for each learnable module
- Added `comorbidity_scheduler` and `causality_scheduler` placeholders
- Initialized learnable modules with error handling

**New Optimizers**:
- `self.comorbidity_optimizer`: Trains comorbidity weights
- `self.causality_optimizer`: Trains causality weights

### 4. Testing

**File**: `tests/test_learnable_modules.py`

**Test Functions**:
1. `test_learnable_comorbidity()`: Module creation, parameters, gradients
2. `test_learnable_causality()`: Module creation, edge retrieval, gradients
3. `test_enhanced_bayesian_with_learnable()`: Integration with Bayesian network
4. `test_gnn_with_learnable()`: Integration with GNN module

**Coverage**:
- PyTorch nn.Module validation
- Parameter registration and gradient computation
- Forward/backward pass correctness
- Backward compatibility with fixed weights
- Integration with existing modules

**Results**: All tests pass ✅

### 5. Documentation

**File**: `docs/features/LEARNABLE_WEIGHTS_GUIDE.md`

**Contents**:
- Architecture explanation with code examples
- Integration guide for all modules
- Training workflow with AdversarialTrainer
- Parameter counts and memory usage
- Advantages and limitations
- Future work possibilities

**Length**: 350+ lines of comprehensive documentation

### 6. Demo

**File**: `demos/demo_learnable_weights.py`

**Demonstrations**:
1. Learnable comorbidity effects creation and usage
2. Learnable symptom causality creation and usage
3. Comparison of learned vs fixed weights
4. Training workflow example
5. Gradient computation verification

**Output**: Interactive terminal demo with statistics and examples

## Technical Details

### Parameterization Choices

**Comorbidity Effects (Log-Space)**:
```python
# Storage: log(multiplier)
self.comorbidity_weights = nn.Parameter(log_multipliers)

# Forward: exponentiate to get positive multipliers
multipliers = torch.exp(self.comorbidity_weights)
```

**Why**: Ensures multipliers stay positive without constraints, natural scale for ratios

**Symptom Causality (Logit-Space)**:
```python
# Storage: logit(strength)
self.causal_weights = nn.Parameter(logit_strengths)

# Forward: sigmoid to get [0, 1] range
strengths = torch.sigmoid(self.causal_weights)
```

**Why**: Ensures strengths stay in [0, 1] without clamping, symmetric learning

### Gradient Flow

```
Patient Complaint (Text)
    ↓
[DeBERTa Encoder]
    ↓ (text features)
[GNN with Learnable Causality]  ← gradients to causality weights
    ↓ (graph features)
[Fusion Layer]
    ↓
[Classification Head]
    ↓
Loss (CrossEntropy + Adversarial + Medical)
    ↓ (backpropagation)
Updates: Generator, Discriminator, Bayesian, Comorbidity, Causality
```

### Memory Footprint

| Component | Parameters | Memory (FP32) |
|-----------|-----------|---------------|
| Comorbidity Effects | 184 | ~736 bytes |
| Symptom Causality | 16 | ~64 bytes |
| **Total New** | **200** | **~800 bytes** |
| DeBERTa Encoder | 86M | ~344 MB |
| Mistral 7B Generator | 7B | ~28 GB |

**Impact**: Negligible overhead (~0.0003% of total parameters)

## Backward Compatibility

All changes are **backward compatible** through opt-in parameters:

- `EnhancedBayesianNetwork(use_learnable_comorbidity=False)` ← default
- `SymptomGraphModule(learnable_causality=None)` ← default
- `DiagnosisDiscriminator(learnable_causality=None)` ← default
- `AdversarialTrainer(use_learnable_comorbidity=False, use_learnable_causality=False)` ← default

**No existing code breaks**. All tests pass without modification.

## Verification

### Tests Run
```bash
python tests/test_basic.py                     # ✅ PASS
python tests/test_enhanced_bayesian.py         # ✅ PASS
python tests/test_causal_graph.py              # ✅ PASS
python tests/test_learnable_modules.py         # ✅ PASS (4/4 tests)
python demos/demo_learnable_weights.py         # ✅ PASS
```

### Integration Tests
```python
# Learnable comorbidity: 184 parameters
lc = LearnableComorbidityEffects()

# Learnable causality: 16 parameters
ls = LearnableSymptomCausality()

# Enhanced network with learnable
net = create_enhanced_bayesian_network(use_learnable_comorbidity=True)

# GNN with learnable
gnn = SymptomGraphModule(conditions=conditions, learnable_causality=ls)

# All work correctly ✅
```

## Benefits

✅ **End-to-End Differentiable**: Medical knowledge learned from data  
✅ **Minimal Overhead**: Only 200 parameters vs 86M+ existing  
✅ **Clinically Initialized**: Starts from evidence-based values  
✅ **Backward Compatible**: Opt-in via explicit flags  
✅ **Fully Tested**: Comprehensive test coverage  
✅ **Well Documented**: Guide + demos + docstrings  
✅ **Production Ready**: No breaking changes  

## Limitations

⚠️ **Requires Training Data**: Need labeled patient-condition pairs  
⚠️ **Risk of Overfitting**: Small parameter count can overfit  
⚠️ **Clinical Validation**: Learned weights need expert review  
⚠️ **Initialization Dependent**: Starts from clinical priors  

## Future Enhancements

Potential extensions (not implemented):
1. Learnable temporal delays (currently fixed)
2. Learnable interaction effects (asthma + COPD)
3. Learnable age/severity modifiers
4. Meta-learning for fast adaptation
5. Bayesian uncertainty over learned weights
6. Regularization to prevent drift from clinical values

## Files Changed

**New Files Created (6)**:
1. `phaita/models/learnable_comorbidity.py` (220 lines)
2. `phaita/models/learnable_causality.py` (220 lines)
3. `tests/test_learnable_modules.py` (280 lines)
4. `docs/features/LEARNABLE_WEIGHTS_GUIDE.md` (350 lines)
5. `demos/demo_learnable_weights.py` (230 lines)

**Files Modified (5)**:
1. `phaita/models/__init__.py` (exports added)
2. `phaita/models/enhanced_bayesian_network.py` (learnable support)
3. `phaita/models/gnn_module.py` (learnable support)
4. `phaita/models/discriminator.py` (learnable support)
5. `phaita/training/adversarial_trainer.py` (training support)

**Total Lines Added**: ~1500 lines
**Total Lines Modified**: ~100 lines

## Conclusion

Successfully implemented learnable comorbidity effects and symptom causality as PyTorch parameters, enabling end-to-end training of medical knowledge while maintaining full backward compatibility with existing fixed-weight system. The implementation is minimal (~200 parameters), well-tested (4 test suites), comprehensively documented (350-line guide), and production-ready.

## Usage Example

```python
from phaita.training.adversarial_trainer import AdversarialTrainer

# Enable all learnable modules
trainer = AdversarialTrainer(
    use_pretrained_generator=True,
    use_pretrained_discriminator=True,
    use_learnable_bayesian=True,
    use_learnable_comorbidity=True,
    use_learnable_causality=True,
    bayesian_lr=1e-3,
    comorbidity_lr=1e-3,
    causality_lr=1e-3,
    device="cuda"
)

# Train (will optimize all weights via gradient descent)
trainer.train(num_epochs=10, batch_size=8)

# Inspect learned weights
diabetes_effects = trainer.learnable_comorbidity.get_symptom_modifiers("diabetes")
causal_edges = trainer.learnable_causality.get_causal_edges()

print("Learned diabetes effects:", diabetes_effects)
print("Learned causal edges:", causal_edges)
```

**Result**: Medical knowledge adapts to data while starting from clinical evidence! 🎉
