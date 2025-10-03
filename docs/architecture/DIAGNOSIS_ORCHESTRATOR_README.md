# Diagnosis Orchestrator with Red-Flag Integration

## Overview

The Diagnosis Orchestrator is a component that combines predictions from multiple sources (Bayesian reasoning and neural networks) and enriches them with red-flag detection and escalation guidance for medical triage.

## Features

- **Weighted Ensemble**: Combines Bayesian priors (from dialogue-based symptom gathering) with neural network predictions (from text-based diagnosis)
- **Red-Flag Detection**: Automatically detects emergency symptoms that require immediate medical attention
- **Three-Tier Escalation**: Classifies cases as emergency, urgent, or routine
- **Clinical Guidance**: Generates appropriate patient guidance for each escalation level

## Files

### Core Implementation
- `phaita/triage/diagnosis_orchestrator.py` - Main orchestrator module
  - `DiagnosisWithContext` - Dataclass for enriched diagnosis predictions
  - `DiagnosisOrchestrator` - Main orchestrator class

### Configuration
- `config/red_flags.yaml` - Red-flag symptom mappings for 10 respiratory conditions

### Testing & Demos
- `test_diagnosis_orchestrator.py` - Comprehensive test suite (11 tests)
- `demo_diagnosis_orchestrator.py` - Interactive demo showing all features

## Usage

### Basic Usage

```python
from phaita.triage import DiagnosisOrchestrator

# Initialize orchestrator
orchestrator = DiagnosisOrchestrator()

# Bayesian probabilities (from dialogue engine)
bayesian_probs = {
    "J45.9": 0.6,   # Asthma
    "J18.9": 0.3,   # Pneumonia
    "J44.9": 0.1,   # COPD
}

# Neural predictions (from discriminator)
neural_predictions = [
    {"condition_code": "J45.9", "condition_name": "Asthma", "probability": 0.7},
    {"condition_code": "J18.9", "condition_name": "Pneumonia", "probability": 0.2},
    {"condition_code": "J44.9", "condition_name": "COPD", "probability": 0.1},
]

# Patient symptoms (for red-flag detection)
patient_symptoms = [
    "severe respiratory distress",  # red-flag!
    "wheezing",
    "cough",
]

# Orchestrate diagnosis
diagnoses = orchestrator.orchestrate_diagnosis(
    bayesian_probs=bayesian_probs,
    neural_predictions=neural_predictions,
    patient_symptoms=patient_symptoms,
    top_k=3
)

# Display results
for diagnosis in diagnoses:
    print(f"{diagnosis.condition_name}: {diagnosis.probability:.1%}")
    print(f"Escalation: {diagnosis.escalation_level}")
    if diagnosis.red_flags:
        print(f"Red-flags: {', '.join(diagnosis.red_flags)}")
    print(f"Guidance: {orchestrator.generate_guidance_text(diagnosis.escalation_level)}")
```

### Integration with Existing Components

```python
from phaita.conversation import DialogueEngine
from phaita.models import DiagnosisDiscriminator
from phaita.triage import DiagnosisOrchestrator

# Initialize components
dialogue_engine = DialogueEngine()
discriminator = DiagnosisDiscriminator(use_pretrained=True)
orchestrator = DiagnosisOrchestrator()

# Get patient complaint
complaint = "I have severe trouble breathing and can't speak full sentences"

# Get neural predictions
neural_preds = discriminator.predict_diagnosis([complaint], top_k=3)[0]

# Get Bayesian probabilities (after dialogue session)
bayesian_probs = dialogue_engine.state.differential_probabilities

# Extract symptoms for red-flag detection
patient_symptoms = [
    "severe respiratory distress",
    "unable to speak full sentences",
]

# Orchestrate combined diagnosis
diagnoses = orchestrator.orchestrate_diagnosis(
    bayesian_probs=bayesian_probs,
    neural_predictions=neural_preds,
    patient_symptoms=patient_symptoms,
    top_k=5
)
```

### Custom Ensemble Weights

```python
# Default: 60% neural, 40% Bayesian
combined = orchestrator.combine_predictions(bayesian_probs, neural_predictions)

# Custom: 30% neural, 70% Bayesian (trust dialogue more)
combined = orchestrator.combine_predictions(
    bayesian_probs,
    neural_predictions,
    neural_weight=0.3,
    bayesian_weight=0.7
)

# Custom: 80% neural, 20% Bayesian (trust text analysis more)
combined = orchestrator.combine_predictions(
    bayesian_probs,
    neural_predictions,
    neural_weight=0.8,
    bayesian_weight=0.2
)
```

## Escalation Logic

The orchestrator uses the following logic to determine escalation levels:

### Emergency
- **Any red-flag symptom is present**, OR
- **High probability (>0.8) for an emergency condition** (J81.0 Acute Pulmonary Edema, J93.0 Pneumothorax)

### Urgent
- **Probability 0.5-0.8 AND no red-flags**

### Routine
- **Probability <0.5 AND no red-flags**

## Red-Flag Configuration

Red-flags are configured in `config/red_flags.yaml`:

```yaml
J45.9:  # Asthma
  red_flags:
    - severe_respiratory_distress
    - unable_to_speak_full_sentences
    - cyanosis
    - altered_mental_status
    - peak_flow_below_50_percent
```

Currently configured for 10 respiratory conditions:
- J45.9 - Asthma
- J18.9 - Pneumonia
- J44.9 - COPD
- J06.9 - Upper Respiratory Infection
- J20.9 - Acute Bronchitis
- J81.0 - Acute Pulmonary Edema
- J93.0 - Spontaneous Pneumothorax
- J15.9 - Bacterial Pneumonia
- J12.9 - Viral Pneumonia
- J21.9 - Acute Bronchiolitis

## Testing

Run the test suite:

```bash
python test_diagnosis_orchestrator.py
```

Test coverage:
- ✅ Ensemble calculation with proper normalization
- ✅ Handling of missing conditions in one source
- ✅ Red-flag detection and matching
- ✅ Escalation level logic (emergency/urgent/routine)
- ✅ Guidance text generation
- ✅ End-to-end orchestration
- ✅ Top-k parameter handling

All 11 tests pass ✅

## Demo

Run the interactive demo:

```bash
python demo_diagnosis_orchestrator.py
```

This demonstrates:
1. Emergency case with red-flags
2. Urgent case without red-flags
3. Routine case with low probabilities
4. Custom ensemble weights

## API Reference

### DiagnosisWithContext

Dataclass representing an enriched diagnosis prediction.

**Attributes:**
- `condition_code` (str): ICD-10 condition code
- `condition_name` (str): Human-readable condition name
- `probability` (float): Combined probability from ensemble
- `red_flags` (List[str]): Detected red-flag symptoms
- `escalation_level` (Literal["emergency", "urgent", "routine"]): Triage urgency
- `reasoning` (str): Explanation of diagnosis and escalation

### DiagnosisOrchestrator

Main orchestrator class.

#### `__init__(red_flags_config_path: Optional[Path] = None)`

Initialize with optional custom red-flags configuration.

#### `combine_predictions(bayesian_probs, neural_predictions, neural_weight=0.6, bayesian_weight=0.4) -> Dict[str, float]`

Combine Bayesian and neural predictions via weighted ensemble.

#### `enrich_with_red_flags(condition_code, patient_symptoms) -> List[str]`

Detect red-flag symptoms for a given condition.

#### `determine_escalation(condition_code, probability, red_flags) -> Literal["emergency", "urgent", "routine"]`

Determine escalation level based on probability and red-flags.

#### `generate_guidance_text(escalation_level) -> str`

Generate patient guidance text for an escalation level.

#### `orchestrate_diagnosis(bayesian_probs, neural_predictions, patient_symptoms, top_k=5) -> List[DiagnosisWithContext]`

Orchestrate complete diagnosis with red-flags and escalation (main method).

## Design Decisions

1. **Weighted Ensemble**: Default 60% neural + 40% Bayesian balances the strengths of both approaches:
   - Neural: Good at pattern recognition in patient text
   - Bayesian: Good at reasoning from confirmed symptoms

2. **Red-Flag Detection**: Uses normalized symptom matching to handle variations in phrasing

3. **Escalation Logic**: Conservative approach that prioritizes patient safety:
   - Any red-flag → emergency (better safe than sorry)
   - High probability → urgent or emergency
   - Low probability → routine monitoring

4. **Normalization**: All probabilities are normalized to sum to 1.0 for consistency

## Future Enhancements

- [ ] Add age-based red-flag thresholds
- [ ] Incorporate vital signs into escalation logic
- [ ] Add severity scoring based on number of red-flags
- [ ] Support custom escalation rules per condition
- [ ] Add confidence intervals to combined probabilities
- [ ] Support temporal progression of symptoms
