# Symptom Normalization Centralization Update

**Date:** 2025-01-04  
**Updated:** 2025-01-XX (Removed all local normalization wrappers)  
**Issue:** Inconsistent symptom normalization across the application  
**Solution:** Centralized normalization functions in `phaita/utils/text.py`

## Latest Update (2025-01-XX)

All local `_normalize_symptom()` wrapper functions have been removed. All files now import and call the centralized functions directly:
- `normalize_symptom()` from `phaita.utils.text` - for spaces format
- `normalize_symptom_to_underscores()` from `phaita.utils.text` - for underscores format

This eliminates 31 lines of redundant wrapper code and ensures true consolidation with no intermediate layers.

## Problem Statement

Previously, symptom normalization was implemented inconsistently across multiple files:

1. **Spaces format** (most common): `symptom.lower().replace('_', ' ').replace('-', ' ').strip()`
   - Used in: `bayesian_network.py`, `dialogue_engine.py`, `diagnosis_orchestrator.py`

2. **Underscores format**: `symptom.strip().lower().replace(' ', '_')`
   - Used in: `info_sheet.py`, `cli.py`, `conversation/engine.py`

3. **Split-only**: `symptom.split('_')`
   - Used in: `discriminator_lite.py`

This inconsistency could lead to:
- Symptoms not matching correctly during probability lookups
- Different behavior in different parts of the system
- Difficult maintenance when normalization logic needs to change

## Solution

Created two centralized normalization functions in `phaita/utils/text.py`:

### 1. `normalize_symptom(symptom: str) -> str`

**Purpose:** Primary normalization for symptom matching and probability lookups  
**Format:** Lowercase with spaces  
**Handles:**
- Converts to lowercase
- Replaces underscores and hyphens with spaces
- Collapses multiple spaces into single space
- Strips leading/trailing whitespace

**Example:**
```python
normalize_symptom('Severe_Respiratory-Distress')
# Returns: 'severe respiratory distress'
```

### 2. `normalize_symptom_to_underscores(symptom: str) -> str`

**Purpose:** Structured data format (info sheets, CLI output)  
**Format:** Lowercase with underscores  
**Handles:**
- Strips whitespace
- Converts to lowercase
- Replaces spaces and hyphens with underscores
- Collapses multiple underscores into single underscore

**Example:**
```python
normalize_symptom_to_underscores('Shortness of Breath')
# Returns: 'shortness_of_breath'
```

## Files Updated

### Core Module Files
1. **phaita/models/bayesian_network.py**
   - Imports `normalize_symptom`
   - Uses centralized function in `get_symptom_probability()`

2. **phaita/models/discriminator_lite.py**
   - Imports `normalize_symptom`
   - Enhanced `_build_symptom_vocabulary()` to properly normalize symptoms

3. **phaita/conversation/dialogue_engine.py**
   - Imports `normalize_symptom`
   - Removed local `_normalize_symptom()` wrapper, calls centralized function directly

4. **phaita/triage/diagnosis_orchestrator.py**
   - Imports `normalize_symptom`
   - Removed local `_normalize_symptom()` wrapper, calls centralized function directly

5. **phaita/triage/info_sheet.py**
   - Imports `normalize_symptom_to_underscores`
   - Removed local `_normalize_symptom()` wrapper, calls centralized function directly

6. **phaita/triage/question_strategy.py**
   - Imports `normalize_symptom`
   - Removed local `_normalise()` function, calls centralized function directly

7. **phaita/conversation/engine.py**
   - Imports `normalize_symptom_to_underscores`
   - `add_symptoms()` uses centralized function

8. **cli.py**
   - Imports `normalize_symptom_to_underscores`
   - Removed local `_normalize_symptom()` wrapper, calls centralized function directly

### New Files
1. **phaita/utils/text.py** - New utility module with normalization functions
2. **tests/test_text_utils.py** - Comprehensive tests for text utilities

## Testing

All existing tests pass with no regressions:

- ✅ `test_basic.py`: 4/4 tests passed
- ✅ `test_text_utils.py`: 3/3 tests passed (NEW)
- ✅ `test_enhanced_bayesian.py`: All tests passed
- ✅ `test_dialogue_engine.py`: 22/22 tests passed
- ✅ `test_diagnosis_orchestrator.py`: 11/11 tests passed
- ✅ `test_conversation_engine.py`: All tests passed
- ✅ `test_conversation_flow.py`: 5/5 tests passed
- ✅ `test_discriminator_lite.py`: All tests passed

### Integration Test Results

Verified that all variants of a symptom name now return identical results:

```python
# BayesianSymptomNetwork
bn.get_symptom_probability('J45.9', 'Shortness_of_Breath')  # 0.8
bn.get_symptom_probability('J45.9', 'shortness-of-breath')  # 0.8
bn.get_symptom_probability('J45.9', 'shortness of breath')  # 0.8
# All three return the same probability ✅
```

## Benefits

1. **Consistency:** All symptom normalization now goes through the same functions
2. **Maintainability:** Changes to normalization logic only need to be made in one place
3. **Reliability:** Symptoms are matched consistently across the entire application
4. **Testability:** Centralized functions are easier to test comprehensively
5. **Documentation:** Clear docstrings explain the purpose and behavior of each format

## Migration Notes

For future development:

- Use `normalize_symptom()` for internal matching and probability lookups
- Use `normalize_symptom_to_underscores()` for structured output and storage
- Import from `phaita.utils.text` instead of implementing local normalization
- Both functions handle edge cases (multiple separators, whitespace, etc.)

## Backward Compatibility

✅ All changes are backward compatible:
- Existing symptom data works without modification
- All tests pass without changes
- API remains the same (internal implementation only)
- No breaking changes to user-facing interfaces

## Related Documentation

- [MODELS_MODULE.md](../modules/MODELS_MODULE.md) - Bayesian network normalization
- [CONVERSATION_MODULE.md](../modules/CONVERSATION_MODULE.md) - Dialogue engine normalization
- [TRIAGE_MODULE.md](../modules/TRIAGE_MODULE.md) - Diagnosis orchestrator normalization

---

**Last Updated:** 2025-01-04  
**Status:** ✅ Complete and tested
