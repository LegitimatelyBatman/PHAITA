# Triage Module Documentation

**Location:** `phaita/triage/`

## Overview

The triage module orchestrates diagnosis generation, red-flag detection, escalation guidance, and information sheet creation for patient care recommendations.

## Components

### 1. Diagnosis Orchestrator (`diagnosis_orchestrator.py`)

**Purpose:** Central hub for generating ranked diagnoses with red-flags and escalation guidance.

**Key Class:** `DiagnosisOrchestrator`

**Features:**
- Generate ranked differential diagnoses (top 10)
- Red-flag detection and severity assessment
- Escalation guidance (emergency vs. routine)
- Confidence scoring
- Evidence integration

**Usage:**
```python
from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator

orchestrator = DiagnosisOrchestrator()

# Generate diagnosis from beliefs
beliefs = {"J45.9": 0.7, "J44.0": 0.2, "J18.9": 0.1}
symptoms = ["shortness_of_breath", "wheezing", "chest_tightness"]

diagnosis_slate = orchestrator.generate_diagnosis_slate(
    beliefs=beliefs,
    symptoms=symptoms,
    age=45,
    comorbidities=["hypertension"]
)

# Display results
for i, dx in enumerate(diagnosis_slate[:3], 1):
    print(f"{i}. {dx['name']} ({dx['confidence']:.1%})")
    if dx['red_flags']:
        print(f"   ⚠️ Red flags: {', '.join(dx['red_flags'])}")
    print(f"   Guidance: {dx['guidance']}")
```

**Output Structure:**
```python
diagnosis_slate = [
    {
        "code": "J45.9",
        "name": "Asthma, unspecified",
        "confidence": 0.70,
        "evidence": ["shortness_of_breath", "wheezing"],
        "red_flags": ["severe_respiratory_distress"],
        "severity": "urgent",
        "guidance": "Seek emergency care immediately",
        "info_sheet_url": "https://..."
    },
    # ... up to 10 diagnoses
]
```

---

### 2. Diagnosis (`diagnosis.py`)

**Purpose:** Individual diagnosis generation and enrichment.

**Key Functions:**
- `generate_diagnosis()` - Create single diagnosis
- `enrich_with_red_flags()` - Add red-flag information
- `calculate_confidence()` - Compute confidence scores

**Usage:**
```python
from phaita.triage.diagnosis import generate_diagnosis, enrich_with_red_flags

# Generate diagnosis
dx = generate_diagnosis(
    condition_code="J45.9",
    belief_probability=0.7,
    supporting_symptoms=["shortness_of_breath", "wheezing"]
)

# Enrich with red flags
dx_enriched = enrich_with_red_flags(
    diagnosis=dx,
    all_symptoms=["shortness_of_breath", "wheezing", "severe_respiratory_distress"]
)

print(f"Red flags: {dx_enriched['red_flags']}")
print(f"Severity: {dx_enriched['severity']}")
```

---

### 3. Info Sheet (`info_sheet.py`)

**Purpose:** Generate patient-friendly information sheets for each diagnosis.

**Key Class:** `InfoSheetGenerator`

**Features:**
- Patient-friendly explanations
- What to expect
- When to seek care
- Self-care recommendations
- Follow-up guidance

**Usage:**
```python
from phaita.triage.info_sheet import InfoSheetGenerator

generator = InfoSheetGenerator()

info_sheet = generator.generate(
    condition_code="J45.9",
    red_flags=["severe_respiratory_distress"],
    patient_age=45
)

print(info_sheet["description"])
print(info_sheet["self_care"])
print(info_sheet["when_to_seek_care"])
```

**Info Sheet Structure:**
```python
{
    "condition_name": "Asthma",
    "description": "Patient-friendly explanation...",
    "what_to_expect": "Timeline and typical course...",
    "self_care": [
        "Use rescue inhaler as prescribed",
        "Avoid triggers",
        "Monitor peak flow"
    ],
    "when_to_seek_care": {
        "emergency": ["Severe difficulty breathing", ...],
        "urgent": ["Increasing symptoms despite treatment", ...],
        "routine": ["Follow up with doctor in 1-2 weeks"]
    },
    "red_flags_present": ["severe_respiratory_distress"],
    "recommended_action": "Seek emergency care immediately"
}
```

---

### 4. Question Strategy (`question_strategy.py`)

**Purpose:** Strategic question selection for efficient triage.

**Key Class:** `QuestionStrategy`

**Features:**
- Discriminative question selection
- Avoid redundant questions
- Clinical reasoning alignment
- Adaptive strategy based on beliefs

**Usage:**
```python
from phaita.triage.question_strategy import QuestionStrategy

strategy = QuestionStrategy()

# Get next best question
next_question = strategy.select_question(
    current_beliefs={"J45.9": 0.6, "J44.0": 0.3, "J18.9": 0.1},
    asked_symptoms=["shortness_of_breath"],
    available_symptoms=["wheezing", "fever", "chest_pain"]
)

print(f"Next question: {next_question['text']}")
print(f"Expected information gain: {next_question['info_gain']:.3f}")
```

**Strategy Types:**
1. **Information Gain:** Maximize entropy reduction (default)
2. **Discriminative:** Separate top conditions
3. **Clinical:** Follow medical decision trees
4. **Hybrid:** Combine multiple strategies

---

## Red-Flag System

### Red-Flag Detection

**Critical Red Flags (Call 911):**
- Severe respiratory distress
- Cyanosis (blue lips/fingers)
- Altered mental status
- Chest pain with radiation
- Respiratory rate > 30 or < 8

**Urgent Red Flags (Seek care within hours):**
- High fever (>103°F)
- Persistent vomiting
- Severe dehydration
- Worsening symptoms

**Concerning (Seek care within 24 hours):**
- Moderate fever (>101°F)
- Persistent cough
- Progressive weakness

### Detection Logic

```python
def detect_red_flags(symptoms: List[str], condition: str) -> List[Dict]:
    """Detect red flags in symptom list."""
    red_flags = []
    
    for symptom in symptoms:
        # Normalize symptom
        normalized = normalize_symptom(symptom)
        
        # Check against red flag registry
        if normalized in RED_FLAG_REGISTRY:
            flag_info = RED_FLAG_REGISTRY[normalized]
            
            # Check if applies to this condition
            if condition in flag_info['conditions'] or 'all' in flag_info['conditions']:
                red_flags.append({
                    'symptom': symptom,
                    'severity': flag_info['severity'],
                    'action': flag_info['action'],
                    'guidance': flag_info['guidance']
                })
    
    return red_flags
```

---

## Escalation Guidance

### Escalation Levels

**Level 1: Emergency (911)**
```python
{
    "level": "emergency",
    "action": "Call 911 or go to ER immediately",
    "reasoning": "Critical symptoms detected",
    "red_flags": ["severe_respiratory_distress", "cyanosis"],
    "time_sensitive": True
}
```

**Level 2: Urgent Care**
```python
{
    "level": "urgent",
    "action": "Seek medical care within 2-4 hours",
    "reasoning": "Urgent symptoms requiring prompt evaluation",
    "red_flags": ["high_fever", "persistent_vomiting"],
    "time_sensitive": True
}
```

**Level 3: Routine Care**
```python
{
    "level": "routine",
    "action": "Schedule appointment with primary care doctor",
    "reasoning": "Non-urgent symptoms",
    "red_flags": [],
    "timeframe": "1-2 weeks"
}
```

### Escalation Decision Tree

```
Symptoms Present
    ↓
Normalize & Match
    ↓
Check Red Flag Registry
    ↓
Any Critical Red Flags? ──Yes──> Level 1: Emergency
    ↓ No
Any Urgent Red Flags? ──Yes──> Level 2: Urgent
    ↓ No
Check Symptom Severity
    ↓
Severe? ──Yes──> Level 2: Urgent
    ↓ No
Level 3: Routine
```

---

## Diagnosis Slate Generation

### Complete Workflow

```python
from phaita.conversation.dialogue_engine import DialogueEngine
from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator

# After conversation concludes
engine = DialogueEngine()
orchestrator = DiagnosisOrchestrator()

# Get final beliefs
beliefs = engine.get_beliefs()
symptoms = engine.get_observed_symptoms()

# Generate slate
diagnosis_slate = orchestrator.generate_diagnosis_slate(
    beliefs=beliefs,
    symptoms=symptoms,
    age=patient_age,
    comorbidities=patient_comorbidities
)

# Check for emergencies
has_emergency = any(dx['severity'] == 'critical' for dx in diagnosis_slate)

if has_emergency:
    print("🚨 EMERGENCY: Call 911 immediately")
    print(f"Reason: {diagnosis_slate[0]['guidance']}")
else:
    # Display top 3 diagnoses
    print("\nMost Likely Diagnoses:")
    for i, dx in enumerate(diagnosis_slate[:3], 1):
        print(f"{i}. {dx['name']} ({dx['confidence']:.1%})")
        
        if dx['red_flags']:
            print(f"   ⚠️ Concerning symptoms: {', '.join(dx['red_flags'])}")
        
        print(f"   Recommended action: {dx['guidance']}")
        print(f"   Learn more: {dx['info_sheet_url']}")
```

---

## Integration Examples

### Example 1: End-to-End Triage

```python
from phaita.conversation.engine import ConversationEngine
from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator

# Initialize
conversation = ConversationEngine()
orchestrator = DiagnosisOrchestrator()

# Conduct conversation
session = conversation.start_conversation(
    initial_complaint="I can't breathe and my chest hurts"
)

while not conversation.should_stop(session):
    question = conversation.get_next_question(session)
    response = get_patient_response(question)
    conversation.process_response(session, question, response)

# Generate diagnosis slate
diagnosis_slate = orchestrator.generate_diagnosis_slate(
    beliefs=session['beliefs'],
    symptoms=list(session['patient_responses'].keys()),
    age=session.get('patient_age', 40)
)

# Handle escalation
if diagnosis_slate[0]['severity'] == 'critical':
    trigger_emergency_alert(diagnosis_slate[0])
else:
    display_diagnosis_slate(diagnosis_slate)
```

### Example 2: Red-Flag Only Check

```python
from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator

orchestrator = DiagnosisOrchestrator()

# Quick red-flag check
symptoms = [
    "chest_pain",
    "shortness_of_breath",
    "severe_respiratory_distress",
    "cyanosis"
]

red_flags = orchestrator.check_red_flags(symptoms)

if red_flags:
    print("⚠️ RED FLAGS DETECTED:")
    for flag in red_flags:
        print(f"  - {flag['symptom']}: {flag['action']}")
    
    # Determine escalation
    max_severity = max(flag['severity'] for flag in red_flags)
    if max_severity == 'critical':
        print("\n🚨 CALL 911 IMMEDIATELY")
```

---

## Testing

**Test Files:**
- `tests/test_diagnosis_orchestrator.py` - Orchestrator tests (11/11)
- `tests/test_escalation_guidance.py` - Escalation logic (6/6)
- `tests/test_end_to_end_triage.py` - Full triage sessions
- `tests/test_triage_differential.py` - Differential diagnosis

**Run Tests:**
```bash
python tests/test_diagnosis_orchestrator.py
python tests/test_escalation_guidance.py
python tests/test_end_to_end_triage.py
```

---

## Configuration

### Triage Configuration (`config.yaml`)

```yaml
triage:
  max_diagnoses: 10
  min_confidence: 0.05
  enable_red_flag_check: true
  enable_info_sheets: true
  escalation_thresholds:
    critical: 0.95
    urgent: 0.80
    routine: 0.50
```

### Red Flags Configuration (`config/red_flags.yaml`)

```yaml
red_flags:
  severe_respiratory_distress:
    severity: critical
    conditions: [all]
    action: "Call 911 immediately"
    guidance: "Life-threatening breathing difficulty"
  
  high_fever:
    severity: urgent
    conditions: [J18.9, J20.9]
    action: "Seek care within 2-4 hours"
    guidance: "May indicate serious infection"
```

---

## Best Practices

### DO:
- ✅ Always check for red flags first
- ✅ Normalize symptoms before matching
- ✅ Generate top 10 differential diagnoses
- ✅ Provide clear escalation guidance
- ✅ Include patient-friendly info sheets

### DON'T:
- ❌ Skip red-flag checks (patient safety critical)
- ❌ Return only single diagnosis (differential is important)
- ❌ Use technical jargon in patient guidance
- ❌ Ignore symptom severity
- ❌ Provide medical advice without disclaimers

---

## Related Documentation

- [CONVERSATION_MODULE.md](CONVERSATION_MODULE.md) - Dialogue engine
- [DATA_MODULE.md](DATA_MODULE.md) - Red-flag definitions
- [docs/architecture/DIAGNOSIS_ORCHESTRATOR_README.md](../architecture/DIAGNOSIS_ORCHESTRATOR_README.md)
- [TESTING.md](../TESTING.md) - Testing guidance

---

**Last Updated:** 2025-01-03
