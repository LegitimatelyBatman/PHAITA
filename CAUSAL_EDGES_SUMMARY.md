# Causal Edges Implementation Summary

## Overview
Successfully implemented causal and temporal relationship edges in PHAITA's symptom graph, enhancing the GNN's ability to model symptom dependencies and progression patterns.

## What Was Implemented

### 1. Configuration System
**File**: `config/symptom_causality.yaml`
- 10 causal edges with clinical evidence references
- 6 temporal edges with typical delay times
- Edge type definitions (co-occurrence=0, causal=1, temporal=2)
- Configurable edge weights and reverse causal factors

### 2. Graph Builder Enhancement
**File**: `phaita/models/gnn_module.py`
- New `build_causal_graph()` method in `SymptomGraphBuilder`
- Automatic loading of causality config from default location
- Merges causal, temporal, and co-occurrence edges
- Creates 3D edge features: [type, strength, delay]
- Directed edges with asymmetric weights (reverse = 30% of forward)

### 3. Graph Attention Network Updates
**File**: `phaita/models/gnn_module.py`
- Added `edge_dim` parameter to `GraphAttentionNetwork`
- All GAT layers now support edge features (edge_dim=3)
- Forward pass accepts `edge_attr` parameter
- Edge features modulate attention weights

### 4. Module Integration
- `SymptomGraphModule`: Added `use_causal_edges` parameter (default True)
- `DiagnosisDiscriminator`: Passes causal edges parameter to GNN
- Backward compatible with `use_causal_edges=False`
- Works with torch.compile optimization

### 5. Testing
**File**: `test_causal_graph.py`
- 6 comprehensive tests covering all functionality
- All tests passing (6/6)
- Validates edge loading, construction, weights, directionality, embeddings, and compatibility

### 6. Demo and Documentation
**Files**: `demo_causal_edges.py`, `CAUSAL_EDGES_GUIDE.md`
- Interactive demo showing causal edges in action
- Comprehensive guide with usage examples
- Documents all causal relationships and configuration

## Key Results

### Graph Statistics
- **Total edges**: 791
  - Co-occurrence: 776 (weight: 0.100-0.400)
  - Causal: 10 (weight: 0.225-0.920)
  - Temporal: 5 (weight: 0.360-0.432)
- **Symptoms**: 58 nodes
- **Edge features**: 3 dimensions per edge
- **Forward/reverse pairs**: 5 with correct asymmetry

### Example Relationships
**Causal**:
- hypoxia → cyanosis (0.92)
- respiratory_distress → tachypnea (0.87)
- chest_tightness → difficulty_breathing (0.80)
- fever → fatigue (0.75)
- hypoxia → confusion (0.78)

**Temporal**:
- fever → fatigue (~24 hours)
- cough → productive_cough (~48 hours)
- shortness_of_breath → severe_distress (~6 hours)

### Impact
- Edge features affect GNN output (mean difference: 0.099-0.175)
- No performance regression
- Memory overhead: +3.09KB
- Backward compatible with existing code

## Testing Status
✅ All tests passing:
- `test_causal_graph.py`: 6/6
- `test_basic.py`: 4/4
- `test_gnn_performance.py`: All benchmarks pass
- `demo_causal_edges.py`: Runs successfully

## Files Changed
**Modified**:
- `phaita/models/gnn_module.py` (major)
- `phaita/models/discriminator.py` (minor)

**Created**:
- `config/symptom_causality.yaml`
- `test_causal_graph.py`
- `demo_causal_edges.py`
- `CAUSAL_EDGES_GUIDE.md`

## Usage Example

```python
from phaita.models.gnn_module import SymptomGraphModule
from phaita.data.icd_conditions import RespiratoryConditions

conditions = RespiratoryConditions.get_all_conditions()

# With causal edges (default)
gnn = SymptomGraphModule(
    conditions=conditions,
    use_causal_edges=True
)

# Without causal edges (backward compatible)
gnn_no_causal = SymptomGraphModule(
    conditions=conditions,
    use_causal_edges=False
)
```

## Benefits
1. **Better Diagnosis**: Models true causal pathways, not just correlations
2. **Temporal Awareness**: Understands symptom progression patterns
3. **Clinical Validity**: Encodes expert medical knowledge
4. **Interpretability**: Attention weights reflect clinical relationships
5. **Backward Compatible**: Can disable for comparison

## Next Steps
To add more causal relationships, edit `config/symptom_causality.yaml`:

```yaml
causal_edges:
  - source: your_source_symptom
    target: your_target_symptom
    strength: 0.85
    evidence: "Clinical reference"
```

Symptoms must exist in `config/respiratory_conditions.yaml`.
