# Comorbidity Modeling Implementation Summary

## Overview
Added comprehensive comorbidity modeling to the Enhanced Bayesian Network, allowing symptom presentations to be influenced by patient comorbidities based on clinical evidence.

## Implementation Details

### 1. Enhanced Bayesian Network Changes
**File**: `phaita/models/enhanced_bayesian_network.py`

#### New Parameter
- Added `comorbidities: Optional[List[str]] = None` parameter to `sample_symptoms()` method
- Fully backward compatible - parameter is optional

#### Comorbidity Modifiers
Loaded from `config/comorbidity_effects.yaml` with fallback defaults:
```python
self.comorbidity_modifiers = {
    "diabetes": {"fatigue": 1.3, "infection_risk": 1.5, ...},
    "hypertension": {"dyspnea": 1.2, "chest_pain": 1.4, ...},
    "obesity": {"shortness_of_breath": 1.5, "exercise_intolerance": 1.3, ...},
    # ... and more
}
```

#### Key Features
1. **Symptom Probability Adjustment**
   - Multiplies base symptom probabilities by comorbidity-specific modifiers
   - Caps all probabilities at 0.95 (configurable via `max_probability`)
   - Effects are multiplicative for multiple comorbidities

2. **Comorbidity-Specific Symptoms**
   - Each comorbidity can add unique symptoms with configurable probability
   - Examples: palpitations (hypertension), sense_of_doom (anxiety)

3. **Cross-Condition Interactions**
   - Special handling for Asthma + COPD = ACOS (Asthma-COPD Overlap Syndrome)
   - Chronic cough probability → 0.9 for ACOS cases
   - Based on GINA/GOLD Guidelines 2023

4. **Priority-Based Trimming**
   - High-probability symptoms (≥0.85) from interactions are preserved
   - Prevents important cross-condition symptoms from being randomly dropped

### 2. Configuration File
**File**: `config/comorbidity_effects.yaml`

#### Comorbidities Defined
- **Cardiovascular**: hypertension, heart_failure
- **Metabolic**: diabetes, obesity
- **Immune**: immunocompromised
- **Psychiatric**: anxiety, depression
- **Respiratory**: copd (for cross-condition interactions)

#### Structure
```yaml
diabetes:
  symptom_modifiers:
    fatigue: 1.3
    infection_risk: 1.5
  specific_symptoms:
    - frequent_urination
    - increased_thirst
  probability: 0.3

interactions:
  asthma_copd:
    conditions: [J45.9]  # Asthma
    comorbidity: copd
    symptom_modifiers:
      chronic_cough: 0.9  # Absolute probability
```

Each entry includes clinical evidence references in comments.

### 3. Comprehensive Tests
**File**: `test_enhanced_bayesian.py`

Added `test_comorbidity_modeling()` function with 6 test cases:
1. ✅ Single comorbidity increases relevant symptom probability
2. ✅ Multiple comorbidities compound effects
3. ✅ Comorbidity-specific symptoms appear
4. ✅ Cross-condition interactions (ACOS)
5. ✅ Comorbidities parameter is optional
6. ✅ Unknown comorbidities handled gracefully

### 4. Demonstration Script
**File**: `demo_comorbidity.py`

Interactive demonstration showing:
- Single vs multiple comorbidity effects
- Comorbidity-specific symptoms
- ACOS interaction (Asthma + COPD)
- Severity + comorbidity combinations
- Real-world clinical scenarios

## Usage Examples

### Basic Usage
```python
from phaita.models.enhanced_bayesian_network import create_enhanced_bayesian_network

network = create_enhanced_bayesian_network()

# Simple case - no comorbidities
symptoms, metadata = network.sample_symptoms("J45.9")

# With comorbidity
symptoms, metadata = network.sample_symptoms(
    "J45.9", 
    comorbidities=["diabetes"]
)

# Multiple comorbidities
symptoms, metadata = network.sample_symptoms(
    "J45.9",
    comorbidities=["diabetes", "obesity", "hypertension"],
    age_group="elderly",
    severity="severe"
)

# ACOS interaction
symptoms, metadata = network.sample_symptoms(
    "J45.9",
    comorbidities=["copd"]  # Triggers ACOS pathway
)
```

### Metadata Structure
```python
metadata = {
    "age_group": "adult",
    "severity": "moderate",
    "presentation_type": "standard",
    "comorbidities": ["diabetes", "obesity"]  # Added when comorbidities present
}
```

## Clinical Evidence Base

All symptom modifiers are based on clinical guidelines and literature:
- **Asthma**: GINA Guidelines 2023
- **COPD**: GOLD Guidelines 2023
- **ACOS**: GINA/GOLD Joint Guidelines
- **Heart Failure**: ESC Guidelines 2021
- **Hypertension**: ESC/ESH Guidelines 2018
- **Diabetes**: ADA Standards 2023

References included as comments in `config/comorbidity_effects.yaml`.

## Backward Compatibility

✅ **Fully backward compatible**
- `comorbidities` parameter is optional (defaults to `None`)
- All existing code continues to work without modification
- Existing tests pass without changes
- CLI can optionally adopt the new parameter but doesn't have to

## Testing Results

```
🧠 Enhanced Bayesian Network Tests
==================================================
✅ Standard sampling works
✅ Rare presentation sampling works
✅ Age-specific sampling works
✅ Severity-specific sampling works
✅ Evidence sources available

🏥 Comorbidity Modeling Tests
==================================================
✅ Single comorbidity modifies probabilities (diabetes: 47/50 fatigue vs 24/50 baseline)
✅ Multiple comorbidities tracked in metadata
✅ Comorbidity-specific symptoms can appear
✅ Cross-condition interactions work (ACOS: 19/20 chronic_cough)
✅ Comorbidities parameter is optional
✅ Unknown comorbidities handled gracefully

🎉 All tests passed!
```

## Files Modified/Created

### Modified
1. `phaita/models/enhanced_bayesian_network.py` - Core implementation
2. `test_enhanced_bayesian.py` - Added comorbidity tests
3. `phaita/models/generator.py` - Made bitsandbytes optional
4. `phaita/models/discriminator.py` - Made torch_geometric optional
5. `phaita/models/question_generator.py` - Made bitsandbytes optional

### Created
1. `config/comorbidity_effects.yaml` - Clinical evidence-based configuration
2. `demo_comorbidity.py` - Interactive demonstration

## Future Enhancements

Potential extensions:
1. Load comorbidity prevalence data for realistic population modeling
2. Add temporal progression (early vs late stage comorbidities)
3. Support for medication effects on symptom presentation
4. Integration with patient demographics for risk stratification
5. Add more cross-condition interactions beyond ACOS

## References

- GINA Guidelines 2023: Global Strategy for Asthma Management and Prevention
- GOLD Guidelines 2023: Global Strategy for Prevention, Diagnosis and Management of COPD
- ESC Heart Failure Guidelines 2021
- ESC/ESH Hypertension Guidelines 2018
- ADA Standards of Medical Care in Diabetes 2023
- CDC Guidelines for Immunocompromised Patients 2022
