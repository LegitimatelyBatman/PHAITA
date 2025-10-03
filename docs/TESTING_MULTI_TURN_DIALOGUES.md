# Multi-Turn Dialogue Integration Tests

This document describes the comprehensive integration tests for PHAITA's multi-turn dialogue triage system.

## Overview

Two new test suites have been added to validate end-to-end triage conversations and escalation guidance:

1. **`test_conversation_flow.py`** - Tests complete triage sessions from initial complaint through diagnosis
2. **`test_escalation_guidance.py`** - Tests red-flag detection and care level routing

## Running the Tests

```bash
# Run conversation flow tests
python test_conversation_flow.py

# Run escalation guidance tests
python test_escalation_guidance.py

# Run all tests together
python test_basic.py && python test_conversation_flow.py && python test_escalation_guidance.py
```

All tests use plain Python (no pytest required) and follow the established PHAITA testing conventions.

## Test Suite 1: Conversation Flow (`test_conversation_flow.py`)

### Purpose
Validates that multi-turn dialogues work correctly from initial symptom through clarifying questions to final diagnosis.

### Tests Included

#### 1. `test_complete_asthma_triage_session()`
**What it tests:** Complete triage workflow for an asthma case
- Initial complaint (wheezing)
- 1-5 clarifying questions based on information gain
- Final differential diagnosis with top 3 conditions
- Asthma correctly appears in top 3

**Expected behavior:**
- System asks relevant clarifying questions
- Bayesian belief updating works correctly
- Terminates when confidence threshold is reached or max turns hit
- Asthma is correctly identified as top diagnosis

#### 2. `test_edge_case_deny_all_symptoms()`
**What it tests:** System behavior when patient denies all symptoms
- User consistently denies all suggested symptoms
- System continues up to max_turns limit
- Eventually terminates without crashing
- Returns a differential (though with low confidence)

**Expected behavior:**
- No crashes or invalid states
- Terminates after exactly max_turns
- All symptoms recorded as denied
- Low confidence in all conditions (appropriate uncertainty)

#### 3. `test_edge_case_conflicting_symptoms()`
**What it tests:** Handling of conflicting symptom evidence
- Patient confirms symptoms from multiple conditions (asthma + pneumonia)
- System handles contradiction gracefully
- Still produces a valid differential

**Expected behavior:**
- No crashes despite conflicting evidence
- Multiple conditions appear in differential
- Bayesian updating handles conflicting priors correctly

#### 4. `test_early_termination_confidence_threshold()`
**What it tests:** Early termination when confidence exceeds threshold
- Strong symptom evidence for asthma
- System terminates before max_turns when confidence > 0.7
- Top diagnosis has high probability (>0.7)

**Expected behavior:**
- Terminates early (< max_turns)
- Top condition has probability > threshold
- Correct diagnosis (asthma) is identified

#### 5. `test_maximum_turn_limit_prevents_infinite_loops()`
**What it tests:** Turn limit enforcement prevents infinite loops
- Very high confidence threshold (0.99) to prevent early termination
- Ambiguous symptoms (alternating yes/no responses)
- System strictly enforces max_turns limit

**Expected behavior:**
- Terminates after exactly max_turns
- No infinite loop occurs
- Turn counter is accurately maintained

### Key Validation Points

- ✅ Turn counting and termination logic
- ✅ Symptom evidence tracking (confirmed/denied)
- ✅ Bayesian belief updating
- ✅ Information gain calculation
- ✅ Differential diagnosis generation
- ✅ Probability normalization and sorting
- ✅ Edge case handling (denials, conflicts, limits)

## Test Suite 2: Escalation Guidance (`test_escalation_guidance.py`)

### Purpose
Validates red-flag detection, escalation level determination, and care routing guidance.

### Tests Included

#### 1. `test_emergency_red_flag_triggers_guidance()`
**What it tests:** Emergency red-flags trigger appropriate escalation
- Patient with asthma symptoms + red-flags (severe respiratory distress, unable to speak)
- Red-flags are detected correctly
- Escalation level set to "emergency"
- Guidance includes emergency keywords (911, immediate, ER)

**Expected behavior:**
- ≥2 red-flags detected for asthma
- Escalation level: emergency
- Guidance mentions 911/emergency room/immediate care

#### 2. `test_urgent_care_routing_moderate_cases()`
**What it tests:** Moderate probability cases route to urgent care
- Moderate probability (0.5-0.8) for asthma
- No red-flag symptoms present
- Routes to "urgent" escalation level
- Guidance recommends 24-48 hour timeframe

**Expected behavior:**
- No red-flags detected
- Escalation level: urgent
- Guidance mentions 24-48 hours or "prompt" care
- No emergency keywords (911, ER)

#### 3. `test_routine_care_guidance_low_probability()`
**What it tests:** Low probability cases route to routine care
- Low probability (<0.5) across all conditions
- No red-flag symptoms
- Routes to "routine" escalation level
- Guidance includes monitoring and self-care advice

**Expected behavior:**
- No red-flags detected
- Escalation level: routine
- Guidance mentions monitoring, rest, hydration
- No urgent or emergency keywords

#### 4. `test_guidance_text_includes_condition_specific_actions()`
**What it tests:** All escalation levels have distinct, actionable guidance
- Emergency: mentions 911, ER, immediate action, serious symptoms
- Urgent: mentions 24-48 hours, doctor appointment, prompt care
- Routine: mentions monitoring, self-care, when to escalate

**Expected behavior:**
- Each level has >50 character descriptive guidance
- Level-specific keywords present
- Actionable recommendations included

#### 5. `test_pneumonia_red_flags_trigger_emergency()`
**What it tests:** Condition-specific red-flags are detected
- Patient with pneumonia symptoms + pneumonia-specific red-flags (confusion, low O2)
- Different red-flags than asthma
- Escalation level: emergency

**Expected behavior:**
- Pneumonia-specific red-flags detected (confusion, oxygen_saturation_below_92)
- Escalation level: emergency
- Demonstrates multi-condition red-flag support

#### 6. `test_high_probability_emergency_condition_without_red_flags()`
**What it tests:** Emergency conditions trigger emergency escalation with high probability
- High probability (>0.8) for J81.0 (Acute Pulmonary Edema)
- No explicit red-flag symptoms listed
- Emergency condition type + high probability → emergency

**Expected behavior:**
- Escalation level: emergency
- Triggered by condition type + probability, not just red-flags
- Demonstrates multiple escalation pathways

### Key Validation Points

- ✅ Red-flag detection and matching
- ✅ Multi-condition red-flag support
- ✅ Escalation level determination logic
- ✅ Emergency/urgent/routine routing
- ✅ Guidance text generation
- ✅ Condition-specific recommendations
- ✅ Probability-based escalation
- ✅ Emergency condition classification

## Integration with Existing Tests

These tests complement existing test suites:

- **`test_dialogue_engine.py`** - Unit tests for DialogueEngine components (belief updating, information gain)
- **`test_diagnosis_orchestrator.py`** - Unit tests for DiagnosisOrchestrator (ensemble, red-flags, escalation)
- **`test_conversation_engine.py`** - Stub-based tests for ConversationEngine logic

The new integration tests validate **end-to-end workflows** using the actual implementations.

## Test Execution Time

- **test_conversation_flow.py**: ~5 seconds (5 tests)
- **test_escalation_guidance.py**: ~3 seconds (6 tests)
- **Total**: ~8 seconds for 11 comprehensive integration tests

## Dependencies

Tests use the following PHAITA components:
- `phaita.conversation.dialogue_engine` - DialogueEngine, DialogueState
- `phaita.triage.diagnosis_orchestrator` - DiagnosisOrchestrator, DiagnosisWithContext
- `phaita.data.icd_conditions` - RespiratoryConditions
- `phaita.models.bayesian_network` - BayesianSymptomNetwork (indirect via DialogueEngine)

No external test frameworks required (pure Python).

## Test Output Format

All tests follow PHAITA conventions:
```
🏥 PHAITA Multi-Turn Dialogue Test Suite
============================================================

🏥 Testing complete asthma triage session...
   ✓ Terminated after 2 clarifying questions
   ✓ Asthma correctly identified as top diagnosis
   ✓ Asked 2 questions: ['sputum_production', 'dyspnea']
   ✓ Top diagnosis: Asthma (P=0.410)
✅ Complete asthma triage session

============================================================
📊 Test Results: 5/5 tests passed
🎉 All conversation flow tests passed!
```

## Coverage Summary

### Conversation Flow Coverage
- ✅ Complete triage sessions (symptom → questions → diagnosis)
- ✅ Multi-turn dialogue progression
- ✅ Question selection via information gain
- ✅ Belief updating with symptom evidence
- ✅ Early termination on confidence threshold
- ✅ Maximum turn limit enforcement
- ✅ Edge cases (deny all, conflicting symptoms)
- ✅ Differential diagnosis generation

### Escalation Guidance Coverage
- ✅ Emergency red-flag detection
- ✅ Urgent care routing (moderate cases)
- ✅ Routine care guidance (low probability)
- ✅ Condition-specific red-flags (asthma, pneumonia)
- ✅ Probability-based escalation
- ✅ Emergency condition classification
- ✅ Guidance text validation
- ✅ Multi-condition support

## Future Enhancements

Potential additions for future test coverage:
- Tests with demographic context (age, comorbidities)
- Tests with medical history context
- Tests for symptom switching/correction
- Performance tests for large symptom sets
- Tests with real patient case data (anonymized)
- Tests for multi-condition comorbidities

## Troubleshooting

### Common Issues

**Issue:** Tests fail with "Should have differential despite conflicts"
- **Cause:** Conflicting symptoms led to numerical instability in Bayesian updates
- **Fix:** Use `min_probability=0.0` in `get_differential_diagnosis()`

**Issue:** Tests fail with "Expected at least N questions"
- **Cause:** System terminated early due to strong evidence
- **Fix:** Adjust confidence threshold or expected question count

**Issue:** Red-flags not detected
- **Cause:** Symptom normalization mismatch (underscores vs spaces)
- **Fix:** Use `_normalize_symptom()` method for comparison

## Related Documentation

- [architecture/DIALOGUE_ENGINE.md](architecture/DIALOGUE_ENGINE.md) - DialogueEngine architecture
- [architecture/DIAGNOSIS_ORCHESTRATOR_README.md](architecture/DIAGNOSIS_ORCHESTRATOR_README.md) - Red-flag system
- [../IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) - Module overview
- [../test_dialogue_engine.py](../test_dialogue_engine.py) - Unit tests
- [../test_diagnosis_orchestrator.py](../test_diagnosis_orchestrator.py) - Unit tests

## Contributing

When adding new integration tests:
1. Follow the established naming convention (`test_*`)
2. Include detailed docstrings explaining what is tested
3. Use emoji indicators for test categories (🏥, 🚑, ⚠️, 📋)
4. Provide detailed assertion messages
5. Print intermediate values for debugging
6. Ensure tests are self-contained (no external dependencies)
7. Keep tests fast (<1 second per test if possible)
