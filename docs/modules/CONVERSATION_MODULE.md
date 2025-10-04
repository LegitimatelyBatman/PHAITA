# Conversation Module Documentation

**Location:** `phaita/conversation/`

## Overview

The conversation module manages multi-turn dialogues with patients, tracks beliefs about diagnoses, generates questions, and orchestrates the triage conversation flow.

## Components

### 1. Dialogue Engine (`dialogue_engine.py`)

**Purpose:** Core conversation engine that manages belief updating and information tracking.

**Key Class:** `DialogueEngine`

**Features:**
- Bayesian belief updating
- Information gain calculation
- Symptom normalization
- Conversation state tracking
- Evidence accumulation

**Usage:**
```python
from phaita.conversation.dialogue_engine import DialogueEngine

# Initialize
engine = DialogueEngine()

# Update beliefs based on symptom
engine.update_beliefs(symptom="shortness_of_breath", present=True)
engine.update_beliefs(symptom="fever", present=False)

# Get current beliefs
beliefs = engine.get_beliefs()
print(f"Top diagnosis: {max(beliefs, key=beliefs.get)}")
print(f"Confidence: {max(beliefs.values()):.2%}")

# Get information gain for potential questions
info_gain = engine.compute_information_gain(symptom="chest_pain")
print(f"Asking about chest_pain would gain {info_gain:.3f} bits")
```

**Belief Updating:**
The engine uses Bayes' rule to update probabilities:
```
P(condition | symptom) = P(symptom | condition) × P(condition) / P(symptom)
```

**Symptom Normalization:**
All symptoms are normalized for consistent matching:
```python
# These all match identically:
engine.update_beliefs("severe_respiratory_distress", True)
engine.update_beliefs("severe respiratory distress", True)
engine.update_beliefs("Severe-Respiratory-Distress", True)
```

**Information Gain:**
Measures how much a question reduces uncertainty:
```
IG(symptom) = H(beliefs_before) - H(beliefs_after)
```
Where H is Shannon entropy.

---

### 2. Conversation Engine (`engine.py`)

**Purpose:** High-level conversation orchestration and flow control.

**Key Class:** `ConversationEngine`

**Features:**
- Multi-turn conversation management
- Question generation and ranking
- Stopping criteria (when to conclude)
- History tracking
- Context preservation

**Usage:**
```python
from phaita.conversation.engine import ConversationEngine

# Initialize
engine = ConversationEngine()

# Start conversation
session = engine.start_conversation(
    initial_complaint="I can't breathe properly"
)

# Conversation loop
while not engine.should_stop(session):
    # Get next question
    question = engine.get_next_question(session)
    print(f"Q: {question['text']}")
    
    # Get patient response (simulated)
    response = get_patient_response(question)
    
    # Update session
    engine.process_response(session, question, response)

# Get final diagnosis
diagnosis = engine.get_diagnosis(session)
print(f"Final diagnosis: {diagnosis}")
```

**Stopping Criteria:**
The engine concludes when:
1. **Confidence threshold:** Top belief > 0.85
2. **Information plateau:** Questions yield < 0.1 bits
3. **Max questions:** Reached question limit (default: 10)
4. **Red flag detected:** Emergency escalation needed

**Session State:**
```python
session = {
    "session_id": "uuid",
    "initial_complaint": "...",
    "asked_symptoms": ["symptom1", "symptom2"],
    "patient_responses": {"symptom1": True, "symptom2": False},
    "beliefs": {"J45.9": 0.7, "J44.0": 0.2, ...},
    "question_count": 5,
    "red_flags": [],
    "created_at": datetime,
    "updated_at": datetime
}
```

---

## Conversation Flow

### Standard Triage Session

```
1. Initial Complaint
   ↓
2. Parse & Extract Symptoms
   ↓
3. Update Initial Beliefs
   ↓
4. Generate Questions (ranked by info gain)
   ↓
5. Ask Top Question
   ↓
6. Process Response
   ↓
7. Update Beliefs
   ↓
8. Check Stopping Criteria
   ├─ Continue → Back to Step 4
   └─ Stop → Step 9
   ↓
9. Generate Diagnosis Slate
   ↓
10. Check Red Flags
    ├─ Red Flag → Emergency Escalation
    └─ No Red Flag → Standard Guidance
    ↓
11. Present Results to Patient
```

---

## Belief Management

### Initialization
```python
# Start with uniform priors
beliefs = {code: 1/10 for code in condition_codes}
```

### Update with Evidence
```python
def update_beliefs(self, symptom: str, present: bool):
    """Update beliefs using Bayes' rule."""
    for condition in self.conditions:
        likelihood = self.get_symptom_probability(condition, symptom)
        
        if not present:
            likelihood = 1 - likelihood
        
        # Bayes update
        self.beliefs[condition] *= likelihood
    
    # Normalize
    total = sum(self.beliefs.values())
    self.beliefs = {k: v/total for k, v in self.beliefs.items()}
```

### Belief Tracking
```python
# Track belief evolution over conversation
history = [
    {"turn": 1, "beliefs": {"J45.9": 0.1, ...}},
    {"turn": 2, "beliefs": {"J45.9": 0.3, ...}},
    {"turn": 3, "beliefs": {"J45.9": 0.7, ...}},
]
```

---

## Question Generation

### Information-Theoretic Approach

Questions are ranked by expected information gain:

```python
def compute_information_gain(self, symptom: str) -> float:
    """Compute expected information gain from asking about symptom."""
    current_entropy = self._compute_entropy(self.beliefs)
    
    # Expected entropy after asking
    expected_entropy = 0.0
    for answer in [True, False]:
        # Compute beliefs if patient answers 'answer'
        hypothetical_beliefs = self._hypothetical_update(symptom, answer)
        posterior_entropy = self._compute_entropy(hypothetical_beliefs)
        
        # Weight by probability of this answer
        prob_answer = self._prob_of_answer(symptom, answer)
        expected_entropy += prob_answer * posterior_entropy
    
    # Information gain
    return current_entropy - expected_entropy
```

### Question Ranking
```python
# Rank all unasked symptoms by information gain
questions = []
for symptom in unasked_symptoms:
    info_gain = engine.compute_information_gain(symptom)
    questions.append({
        "symptom": symptom,
        "text": format_question(symptom),
        "info_gain": info_gain
    })

# Sort by information gain (descending)
questions.sort(key=lambda q: q["info_gain"], reverse=True)

# Return top question
best_question = questions[0]
```

---

## Integration Examples

### Example 1: Basic Conversation

```python
from phaita.conversation.dialogue_engine import DialogueEngine
from phaita.triage.diagnosis_orchestrator import DiagnosisOrchestrator

engine = DialogueEngine()
orchestrator = DiagnosisOrchestrator()

# Initial complaint
complaint = "I've been having trouble breathing"
initial_symptoms = extract_symptoms(complaint)

for symptom in initial_symptoms:
    engine.update_beliefs(symptom, present=True)

# Ask questions until confident
while max(engine.get_beliefs().values()) < 0.85:
    # Find best question
    unasked = [s for s in all_symptoms if s not in engine.asked_symptoms]
    best_symptom = max(unasked, key=engine.compute_information_gain)
    
    # Ask (simulated)
    print(f"Do you have {best_symptom}?")
    response = input("(yes/no): ").lower() == "yes"
    
    # Update
    engine.update_beliefs(best_symptom, present=response)

# Final diagnosis
beliefs = engine.get_beliefs()
top_condition = max(beliefs, key=beliefs.get)
diagnosis = orchestrator.generate_diagnosis(top_condition, beliefs[top_condition])

print(f"\nDiagnosis: {diagnosis['name']}")
print(f"Confidence: {diagnosis['confidence']:.1%}")
if diagnosis['red_flags']:
    print("⚠️ RED FLAGS DETECTED - SEEK IMMEDIATE CARE")
```

### Example 2: Multi-Turn with History

```python
from phaita.conversation.engine import ConversationEngine

engine = ConversationEngine()

# Start
session = engine.start_conversation(
    initial_complaint="I have chest pain and I'm short of breath"
)

# Conversation with history
for turn in range(10):
    if engine.should_stop(session):
        break
    
    # Get question
    question = engine.get_next_question(session)
    
    # Simulate response
    response = simulate_patient_response(question)
    
    # Update
    engine.process_response(session, question, response)
    
    # Log
    print(f"Turn {turn+1}:")
    print(f"  Q: {question['text']}")
    print(f"  A: {response}")
    print(f"  Top belief: {max(session['beliefs'].values()):.2%}")

# Result
diagnosis = engine.get_diagnosis(session)
print(f"\nFinal diagnosis: {diagnosis}")
```

---

## Testing

**Test Files:**
- `tests/test_dialogue_engine.py` - Belief updating, info gain
- `tests/test_conversation_engine.py` - Flow control
- `tests/test_conversation_flow.py` - End-to-end sessions

**Run Tests:**
```bash
python tests/test_dialogue_engine.py
python tests/test_conversation_engine.py
python tests/test_conversation_flow.py
```

---

## Configuration

### Conversation Parameters (`config.yaml`)

```yaml
conversation:
  max_questions: 10
  confidence_threshold: 0.85
  min_info_gain: 0.1
  enable_red_flag_escalation: true
  track_history: true
```

---

## Best Practices

### DO:
- ✅ Normalize symptoms before matching
- ✅ Track conversation history for context
- ✅ Check red flags at every turn
- ✅ Use information gain for question ranking
- ✅ Set reasonable stopping criteria

### DON'T:
- ❌ Ask the same question twice
- ❌ Ignore low information gain (don't ask useless questions)
- ❌ Skip symptom normalization
- ❌ Continue past confidence threshold
- ❌ Forget to check for red flags

---

## Advanced Features

### 1. Context-Aware Questions
Questions consider conversation context:
```python
# If patient mentioned "worse at night"
question = "Do your symptoms improve during the day?"
```

### 2. Follow-Up Clarifications
```python
# If patient said "chest pain"
follow_up = "Can you describe the chest pain? (sharp, dull, burning)"
```

### 3. Temporal Reasoning
```python
# Ask about symptom timeline
question = "How long have you had these symptoms?"
# Use temporal module to interpret
```

---

## Related Documentation

- [TRIAGE_MODULE.md](TRIAGE_MODULE.md) - Diagnosis orchestration
- [MODELS_MODULE.md](MODELS_MODULE.md) - Bayesian networks
- [docs/architecture/DIALOGUE_ENGINE.md](../architecture/DIALOGUE_ENGINE.md) - Architecture details
- [TESTING_MULTI_TURN_DIALOGUES.md](../TESTING_MULTI_TURN_DIALOGUES.md) - Test examples

---

**Last Updated:** 2025-01-03
