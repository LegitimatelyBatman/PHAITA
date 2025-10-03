# PHAITA Documentation Index

Complete guide to all PHAITA documentation.

## Core Documentation (Start Here)

### [README.md](../README.md)
**Quick start guide** - Overview, installation, CLI recipes, and basic usage.
- System requirements (GPU, dependencies, models)
- Architecture snapshot
- Getting started commands
- Python API examples
- **Start here for new users**

### [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)
**Problem statement and vision** - Why PHAITA exists and where it's going.
- Medical triage problem statement
- Synthetic data approach
- Akinator-style conversation vision
- Project goals and scope

### [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)
**High-level architecture** - Tour of major modules and how they fit together.
- Module organization (data, models, training, utils)
- Component descriptions
- Data flow diagrams
- Key exports and APIs

### [CHANGE_HISTORY.md](../CHANGE_HISTORY.md)
**Project evolution** - Historical fixes, audits, and outstanding work.
- Final product vision
- Critical fixes completed
- Deep-learning transformation highlights
- Recent audit findings
- Outstanding priorities

## Feature Documentation

### Dialogue and Conversation

#### [docs/architecture/DIALOGUE_ENGINE.md](architecture/DIALOGUE_ENGINE.md)
**Multi-turn dialogue system** - Architecture of the question-asking conversation engine.
- DialogueEngine components
- Belief updating mechanism
- Information gain calculation
- State management across turns
- Integration with Bayesian networks

#### [docs/architecture/DIAGNOSIS_ORCHESTRATOR_README.md](architecture/DIAGNOSIS_ORCHESTRATOR_README.md)
**Red-flag integration** - How diagnosis orchestration works with emergency detection.
- DiagnosisOrchestrator class
- Red-flag symptom mappings (config/red_flags.yaml)
- Escalation logic (emergency/urgent/routine)
- Ensemble prediction (Bayesian + neural)
- Guidance text generation
- **11 comprehensive tests included**

#### [docs/architecture/IMPLEMENTATION_DIALOGUE_ENGINE.md](architecture/IMPLEMENTATION_DIALOGUE_ENGINE.md)
**Detailed dialogue implementation** - Deep dive into dialogue engine internals.
- Implementation details
- Algorithm descriptions
- Design decisions

### Deep Learning and Models

#### [DEEP_LEARNING_GUIDE.md](../DEEP_LEARNING_GUIDE.md)
**GPU setup and troubleshooting** - Practical guide for enabling full model stack.
- CUDA setup
- Model downloading from HuggingFace
- Memory optimization
- 4-bit quantization with bitsandbytes
- Troubleshooting common issues

#### [IMPLEMENTATION_DETAILS.md](../IMPLEMENTATION_DETAILS.md)
**Deep-learning architecture** - Technical details of neural network components.
- DeBERTa encoder specifications
- Graph neural network architecture
- Mistral 7B integration
- Model parameters and sizes
- Training procedures

#### [docs/features/MODEL_LOADER_GUIDE.md](features/MODEL_LOADER_GUIDE.md)
**Model loading utilities** - How model initialization and caching works.
- Model loader API
- Quantization support
- Device management
- Caching strategies
- Error handling

### Graph Neural Networks

#### [docs/features/GNN_QUICK_REFERENCE.md](features/GNN_QUICK_REFERENCE.md)
**GNN module quick guide** - Fast reference for symptom graph features.
- SymptomGraphBuilder usage
- SymptomGraphModule API
- Performance tips
- Example code snippets

#### [docs/features/GNN_OPTIMIZATION_RESULTS.md](features/GNN_OPTIMIZATION_RESULTS.md)
**GNN performance benchmarks** - Optimization results and comparisons.
- Performance metrics
- Memory usage
- Speed improvements
- Benchmark results

#### [docs/features/CAUSAL_EDGES_GUIDE.md](features/CAUSAL_EDGES_GUIDE.md)
**Causal relationships in symptom graphs** - Complete guide to causal edge implementation.
- Causal edge types
- Configuration (config/symptom_causality.yaml)
- Usage examples
- Clinical rationale for relationships

#### [docs/features/CAUSAL_EDGES_SUMMARY.md](features/CAUSAL_EDGES_SUMMARY.md)
**Causal edges implementation summary** - What was implemented and key results.
- Implementation overview
- Graph statistics
- Testing status (6/6 tests pass)
- Files changed

### Advanced Features

#### [docs/features/TEMPORAL_MODELING_README.md](features/TEMPORAL_MODELING_README.md)
**Symptom progression over time** - Temporal modeling of symptom evolution.
- SymptomTimeline tracking
- LSTM-based temporal encoder
- Temporal pattern matching
- Integration with diagnosis

#### [docs/features/COMORBIDITY_IMPLEMENTATION.md](features/COMORBIDITY_IMPLEMENTATION.md)
**Comorbidity effects** - How chronic conditions affect symptom presentation.
- Comorbidity modifiers (diabetes, hypertension, obesity, etc.)
- Interaction effects (ACOS, heart failure + COPD)
- Clinical evidence references
- Configuration (config/comorbidity_effects.yaml)

#### [docs/features/IMPLEMENTATION_CHECKLIST.md](features/IMPLEMENTATION_CHECKLIST.md)
**Comorbidity implementation checklist** - Detailed implementation tracking.
- Problem statement requirements
- File changes
- Test results
- Verification status

### Template System

#### [docs/features/TEMPLATE_IMPLEMENTATION.md](features/TEMPLATE_IMPLEMENTATION.md)
**Template-based generation** - Detailed template system architecture.
- 28 template patterns
- Intelligent selection algorithm
- Age/severity/formality factors
- Diversity metrics

#### [docs/features/TEMPLATE_QUICKSTART.md](features/TEMPLATE_QUICKSTART.md)
**Template system quick start** - Fast guide to template generation.
- Quick usage examples
- Template categories
- Performance metrics (81.3% uniqueness)
- Running tests

## Testing Documentation

### [TESTING.md](TESTING.md) ⭐ **NEW**
**Comprehensive testing guide** - Complete reference for all 22 test files.
- All test files documented with purpose and timings
- Test categories (core, dialogue, models, CLI, integration)
- Test execution strategies
- Dependency requirements
- Troubleshooting common issues
- Contributing new tests
- **Start here for testing**

### [TESTING_MULTI_TURN_DIALOGUES.md](TESTING_MULTI_TURN_DIALOGUES.md)
**Dialogue integration tests** - Detailed coverage of conversation flow tests.
- test_conversation_flow.py (5 tests)
- test_escalation_guidance.py (6 tests)
- Test descriptions and expected behaviors
- Coverage summary
- Integration with existing tests

## Configuration Files

### Respiratory Conditions
- **config/respiratory_conditions.yaml** - 10 ICD-10 respiratory conditions with symptoms, severity indicators, lay terms
- Environment variable: `PHAITA_RESPIRATORY_CONFIG`
- Hot-reload supported: `RespiratoryConditions.reload()`

### Red-Flags
- **config/red_flags.yaml** - Emergency symptom mappings for all 10 conditions
- Used by DiagnosisOrchestrator for escalation level determination

### Symptom Causality
- **config/symptom_causality.yaml** - Causal and temporal edges between symptoms
- Used by SymptomGraphBuilder for GNN construction

### Comorbidity Effects
- **config/comorbidity_effects.yaml** - 8 comorbidities with symptom modifiers and interactions
- Used by EnhancedBayesianNetwork for realistic symptom modulation

### Templates
- **phaita/data/templates.yaml** - 28 complaint generation templates
- Metadata: age ranges, severity, formality, symptom count

## Quick Navigation by Task

### I want to...

#### Get started with PHAITA
1. [README.md](../README.md) - Installation and quick start
2. [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md) - Understand the vision
3. [TESTING.md](TESTING.md) - Run tests to verify installation

#### Understand the architecture
1. [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) - High-level overview
2. [IMPLEMENTATION_DETAILS.md](../IMPLEMENTATION_DETAILS.md) - Deep-learning details
3. [docs/architecture/DIALOGUE_ENGINE.md](architecture/DIALOGUE_ENGINE.md) - Conversation system

#### Set up deep learning models
1. [DEEP_LEARNING_GUIDE.md](../DEEP_LEARNING_GUIDE.md) - GPU setup and troubleshooting
2. [docs/features/MODEL_LOADER_GUIDE.md](features/MODEL_LOADER_GUIDE.md) - Model loading utilities
3. [README.md](../README.md) - System requirements

#### Work with the dialogue system
1. [docs/architecture/DIALOGUE_ENGINE.md](architecture/DIALOGUE_ENGINE.md) - Architecture and API
2. [docs/architecture/DIAGNOSIS_ORCHESTRATOR_README.md](architecture/DIAGNOSIS_ORCHESTRATOR_README.md) - Red-flags and escalation
3. [TESTING_MULTI_TURN_DIALOGUES.md](TESTING_MULTI_TURN_DIALOGUES.md) - Test examples

#### Customize medical knowledge
1. Edit: `config/respiratory_conditions.yaml`
2. Edit: `config/red_flags.yaml`
3. Edit: `config/comorbidity_effects.yaml`
4. Edit: `config/symptom_causality.yaml`

#### Add new features
1. [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) - Module organization
2. [TESTING.md](TESTING.md) - Adding new tests
3. [CHANGE_HISTORY.md](../CHANGE_HISTORY.md) - Outstanding priorities

#### Troubleshoot issues
1. [DEEP_LEARNING_GUIDE.md](../DEEP_LEARNING_GUIDE.md) - GPU and model issues
2. [TESTING.md](TESTING.md) - Test failures and common issues
3. [README.md](../README.md) - System requirements and setup

#### Run tests
1. [TESTING.md](TESTING.md) - **Complete testing guide (22 test files)**
2. [TESTING_MULTI_TURN_DIALOGUES.md](TESTING_MULTI_TURN_DIALOGUES.md) - Dialogue tests
3. Run: `python test_basic.py` - Start here

## Documentation by File Location

### Root Directory
- README.md
- PROJECT_SUMMARY.md
- IMPLEMENTATION_SUMMARY.md
- IMPLEMENTATION_DETAILS.md
- CHANGE_HISTORY.md
- DEEP_LEARNING_GUIDE.md

### docs/ Directory

#### Core Documentation
- docs/DOCUMENTATION_INDEX.md ⭐ **NEW** - This file
- docs/TESTING.md ⭐ **NEW** - Comprehensive testing guide
- docs/TESTING_MULTI_TURN_DIALOGUES.md - Dialogue integration tests

#### Architecture Documentation (docs/architecture/)
- docs/architecture/DIALOGUE_ENGINE.md
- docs/architecture/IMPLEMENTATION_DIALOGUE_ENGINE.md
- docs/architecture/DIAGNOSIS_ORCHESTRATOR_README.md

#### Feature Documentation (docs/features/)
- docs/features/MODEL_LOADER_GUIDE.md
- docs/features/GNN_QUICK_REFERENCE.md
- docs/features/GNN_OPTIMIZATION_RESULTS.md
- docs/features/CAUSAL_EDGES_GUIDE.md
- docs/features/CAUSAL_EDGES_SUMMARY.md
- docs/features/TEMPORAL_MODELING_README.md
- docs/features/COMORBIDITY_IMPLEMENTATION.md
- docs/features/IMPLEMENTATION_CHECKLIST.md
- docs/features/TEMPLATE_IMPLEMENTATION.md
- docs/features/TEMPLATE_QUICKSTART.md

### .github/ Directory
- .github/copilot-instructions.md - Instructions for GitHub Copilot

## Document Status

### Core Documentation
- ✅ README.md - Up to date
- ✅ PROJECT_SUMMARY.md - Current
- ✅ IMPLEMENTATION_SUMMARY.md - Current
- ✅ CHANGE_HISTORY.md - Needs update (references pytest, should be plain Python)

### Testing Documentation
- ✅ TESTING.md - **NEW** comprehensive guide
- ✅ TESTING_MULTI_TURN_DIALOGUES.md - Current
- ⚠️ .github/copilot-instructions.md - Needs update with full test list

### Feature Documentation
- ✅ All feature docs current and accurate

## Contributing to Documentation

When adding new documentation:

1. **Location:**
   - Core guides → root directory
   - Testing guides → docs/
   - Feature-specific → root with descriptive name

2. **Style:**
   - Use markdown headers (##, ###)
   - Include code examples with ```bash or ```python
   - Add emoji for visual clarity (✅, ⚠️, 🧪, etc.)
   - Cross-reference related docs with [text](path)

3. **Updates:**
   - Update this index when adding new docs
   - Update README.md "Documentation Map" section
   - Add cross-references in related docs

4. **Naming:**
   - Use UPPERCASE for documentation files
   - Use snake_case for config/data files
   - Be descriptive: FEATURE_GUIDE.md not GUIDE.md

## Questions?

- Check README.md for quick start
- Check TESTING.md for test issues
- Check DEEP_LEARNING_GUIDE.md for GPU/model issues
- Check IMPLEMENTATION_SUMMARY.md for architecture questions
- Review relevant feature documentation above

## License

PHAITA is released under the MIT License. For research and educational purposes only - not for clinical use without regulatory approval.
