# Dialogue Engine with Bayesian Belief Updating

## Overview

The `DialogueEngine` implements a Bayesian inference system for medical triage conversations. It maintains probability distributions over possible diagnoses and updates beliefs using Bayes' rule as symptom evidence is gathered through questions.

## Key Features

- **Bayesian Belief Updating**: Uses Bayes' rule to update condition probabilities based on symptom evidence
- **Information Gain Question Selection**: Selects the most informative questions using entropy reduction
- **Intelligent Termination**: Stops asking questions when confidence is high enough
- **Question Repetition Prevention**: Tracks asked questions to avoid redundancy
- **Differential Diagnosis**: Provides ranked list of possible conditions with confidence scores

## Architecture

### DialogueState

Tracks the state of the conversation:

```python
@dataclass
class DialogueState:
    differential_probabilities: Dict[str, float]  # P(condition) for each condition
    asked_questions: List[str]                     # History of asked symptoms
    confirmed_symptoms: Set[str]                   # Symptoms patient has
    denied_symptoms: Set[str]                      # Symptoms patient doesn't have
    turn_count: int                                # Number of turns taken
    confidence_threshold: float = 0.7              # Threshold for termination
```

### DialogueEngine

Main conversation engine with Bayesian inference:

```python
engine = DialogueEngine(
    conditions=None,              # Uses RespiratoryConditions by default
    initial_prior=None,           # Uniform prior by default
    max_turns=10,                 # Maximum conversation length
    confidence_threshold=0.7      # Minimum confidence for termination
)
```

## Usage Example

### Basic Conversation Flow

```python
from phaita import DialogueEngine

# Initialize the engine
engine = DialogueEngine(max_turns=10, confidence_threshold=0.7)

# Conduct conversation
while not engine.should_terminate():
    # Select most informative question
    symptom = engine.select_next_question()
    if symptom is None:
        break
    
    # Ask patient about symptom
    print(f"Do you have {symptom}?")
    answer = input("(yes/no): ")
    
    # Update beliefs based on answer
    present = answer.lower() == "yes"
    engine.answer_question(symptom, present)
    
    # Get current differential
    differential = engine.get_differential_diagnosis(top_n=3)
    for entry in differential:
        print(f"  {entry['name']}: {entry['probability']:.1%}")

# Get final diagnosis
final_diagnosis = engine.get_differential_diagnosis(top_n=5)
```

## Methods

### update_beliefs(symptom: str, present: bool)

Updates condition probabilities using Bayes' rule:

```
P(condition | symptom) ∝ P(symptom | condition) × P(condition)
```

**Arguments:**
- `symptom`: Name of the symptom (e.g., "wheezing")
- `present`: True if symptom is present, False if absent

**Example:**
```python
engine.update_beliefs("wheezing", present=True)
engine.update_beliefs("fever", present=False)
```

### should_terminate() -> bool

Determines if conversation should end. Returns True if:
- Top condition probability > `confidence_threshold` (default 0.7)
- Top 3 conditions sum to > 0.9
- Turn count >= `max_turns`

**Example:**
```python
if engine.should_terminate():
    print("Enough information gathered")
```

### select_next_question() -> Optional[str]

Selects the most informative symptom to ask about using information gain:

```
IG(symptom) = H(before) - E[H(after)]
```

Where H is Shannon entropy and E[H(after)] is expected entropy after learning the symptom status.

**Returns:** Symptom name with highest information gain, or None if all asked

**Example:**
```python
next_symptom = engine.select_next_question()
if next_symptom:
    print(f"Ask about: {next_symptom}")
```

### get_differential_diagnosis(top_n=10, min_probability=0.01) -> List[Dict]

Returns ranked differential diagnosis.

**Arguments:**
- `top_n`: Maximum number of conditions to return
- `min_probability`: Filter out conditions below this threshold

**Returns:** List of dicts with keys:
- `condition_code`: ICD-10 code
- `name`: Condition name
- `probability`: Current probability

**Example:**
```python
differential = engine.get_differential_diagnosis(top_n=5, min_probability=0.05)
for entry in differential:
    print(f"{entry['name']}: {entry['probability']:.1%}")
```

### answer_question(symptom: str, present: bool)

Convenience method that updates beliefs and increments turn counter.

**Example:**
```python
engine.answer_question("wheezing", present=True)
```

### reset()

Resets the engine to initial state with uniform priors.

**Example:**
```python
engine.reset()  # Start a new conversation
```

## Bayesian Inference Details

### Likelihood Ratios

The engine uses likelihood ratios from the `BayesianSymptomNetwork`:
- Primary symptoms: P = 0.8
- Severity indicators: P = 0.4
- Absent symptoms: P = 1 - P(symptom | condition)

### Probability Normalization

After each update, probabilities are normalized to ensure they sum to 1.0:

```python
P(condition) = P(condition) / Σ P(all_conditions)
```

### Information Gain

Information gain measures expected reduction in uncertainty:

1. Calculate current entropy: `H = -Σ P(c) log₂ P(c)`
2. Calculate expected entropy if symptom is present
3. Calculate expected entropy if symptom is absent
4. Weight by probability of each scenario
5. Information gain = current entropy - expected entropy

## Termination Criteria

The engine uses three criteria for deciding when to stop:

1. **High Confidence**: Top condition > 0.7 probability
2. **Clear Top 3**: Top 3 conditions sum to > 0.9 (differential is clear)
3. **Turn Limit**: Reached maximum turns (safety limit)

This prevents infinite loops while ensuring sufficient information is gathered.

## Integration with Existing Systems

The `DialogueEngine` complements the existing `ConversationEngine`:

- **ConversationEngine**: Manages conversation flow, question generation, and state tracking
- **DialogueEngine**: Provides Bayesian reasoning, information gain calculation, and termination logic

Both can be used together for a complete triage system.

## Testing

Run the comprehensive test suite:

```bash
python test_dialogue_engine.py
```

Tests cover:
- Belief updating with positive/negative evidence
- Termination conditions
- Question repetition prevention
- Information gain calculation
- Differential diagnosis generation
- State management

Run the demo:

```bash
python demo_dialogue_engine.py
```

## Performance Characteristics

- **Memory**: O(n) where n is number of conditions (~10 for respiratory)
- **Time per update**: O(n) for Bayesian update
- **Time per question selection**: O(s) where s is number of symptoms (~50 for respiratory)
- **Typical conversation length**: 3-6 turns for high-confidence diagnosis

## Future Enhancements

Potential improvements:
- [ ] Multi-symptom questions (e.g., "Do you have A or B?")
- [ ] Temporal reasoning (symptom progression over time)
- [ ] Integration with patient history/demographics
- [ ] Active learning to improve likelihood estimates
- [ ] Explanation generation for differential diagnosis

## References

- Bayesian inference in medical diagnosis
- Information theory and entropy
- Decision theory for optimal question selection
- Medical triage protocols
