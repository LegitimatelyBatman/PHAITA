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

### [INSTALLATION.md](../INSTALLATION.md)
**Detailed installation guide** - Complete instructions for all installation methods.
- Modular installation options (CPU-only, GPU, development, scraping)
- Requirements files explanation (base, gpu, dev, scraping)
- Use case-specific installation instructions
- Troubleshooting and compatibility notes
- Migration guide from old installation method

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

#### [docs/features/GNN_OPTIMIZATION_GUIDE.md](features/GNN_OPTIMIZATION_GUIDE.md)
**GNN optimization complete guide** - Comprehensive guide to GNN performance and usage.
- Quick start and usage examples
- Performance benchmarks (29.6% speedup achieved)
- Key optimizations (cached buffers, torch.compile)
- Profiling results and bottleneck analysis
- Memory efficiency metrics
- Platform-specific performance expectations
- Testing and validation
- Troubleshooting guide

#### [docs/features/CAUSAL_EDGES_GUIDE.md](features/CAUSAL_EDGES_GUIDE.md)
**Causal relationships in symptom graphs** - Complete guide to causal edge implementation.
- Implementation overview and statistics
- Causal edge types (co-occurrence, causal, temporal)
- Configuration (config/symptom_causality.yaml)
- Usage examples and integration
- Clinical rationale for relationships
- Testing status (6/6 tests pass)
- Files changed and implementation details

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
2. [docs/guides/SOP.md](guides/SOP.md) - Complete Standard Operating Procedure
3. [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md) - Understand the vision
4. [TESTING.md](TESTING.md) - Run tests to verify installation

#### Understand the architecture
1. [docs/modules/IMPLEMENTATION_SUMMARY.md](modules/IMPLEMENTATION_SUMMARY.md) - High-level overview
2. [docs/modules/IMPLEMENTATION_DETAILS.md](modules/IMPLEMENTATION_DETAILS.md) - Deep-learning details
3. [docs/modules/DATA_MODULE.md](modules/DATA_MODULE.md) - Data layer
4. [docs/modules/MODELS_MODULE.md](modules/MODELS_MODULE.md) - Neural networks
5. [docs/modules/CONVERSATION_MODULE.md](modules/CONVERSATION_MODULE.md) - Dialogue system
6. [docs/modules/TRIAGE_MODULE.md](modules/TRIAGE_MODULE.md) - Diagnosis and red-flags

#### Set up deep learning models
1. [DEEP_LEARNING_GUIDE.md](../DEEP_LEARNING_GUIDE.md) - GPU setup and troubleshooting
2. [docs/features/MODEL_LOADER_GUIDE.md](features/MODEL_LOADER_GUIDE.md) - Model loading utilities
3. [docs/guides/SOP.md](guides/SOP.md) - Training procedures

#### Train models
1. [docs/guides/SOP.md](guides/SOP.md) - Complete training guide
2. [DEEP_LEARNING_GUIDE.md](../DEEP_LEARNING_GUIDE.md) - GPU requirements
3. [docs/modules/MODELS_MODULE.md](modules/MODELS_MODULE.md) - Model architecture

#### Work with the dialogue system
1. [docs/modules/CONVERSATION_MODULE.md](modules/CONVERSATION_MODULE.md) - Dialogue engine guide
2. [docs/architecture/DIALOGUE_ENGINE.md](architecture/DIALOGUE_ENGINE.md) - Architecture details
3. [docs/modules/TRIAGE_MODULE.md](modules/TRIAGE_MODULE.md) - Diagnosis orchestration
4. [TESTING_MULTI_TURN_DIALOGUES.md](TESTING_MULTI_TURN_DIALOGUES.md) - Test examples

#### Customize medical knowledge
1. [docs/guides/PHYSICIAN_CONFIGURATION_GUIDE.md](guides/PHYSICIAN_CONFIGURATION_GUIDE.md) - Step-by-step guide for physicians
2. Edit: `config/respiratory_conditions.yaml`
3. Edit: `config/red_flags.yaml`
4. Edit: `config/comorbidity_effects.yaml`
5. Edit: `config/symptom_causality.yaml`
6. See: [docs/modules/DATA_MODULE.md](modules/DATA_MODULE.md) - Configuration guide

#### Add new features
1. [docs/modules/IMPLEMENTATION_SUMMARY.md](modules/IMPLEMENTATION_SUMMARY.md) - Module organization
2. [TESTING.md](TESTING.md) - Adding new tests
3. [CHANGE_HISTORY.md](../CHANGE_HISTORY.md) - Outstanding priorities
4. [docs/updates/UPDATE_LOG.md](updates/UPDATE_LOG.md) - Recent updates

#### Troubleshoot issues
1. [docs/guides/SOP.md](guides/SOP.md) - Troubleshooting section
2. [DEEP_LEARNING_GUIDE.md](../DEEP_LEARNING_GUIDE.md) - GPU and model issues
3. [TESTING.md](TESTING.md) - Test failures and common issues
4. [README.md](../README.md) - System requirements and setup

#### Run tests
1. [TESTING.md](TESTING.md) - **Complete testing guide (26 test files)**
2. [TESTING_MULTI_TURN_DIALOGUES.md](TESTING_MULTI_TURN_DIALOGUES.md) - Dialogue tests
3. Run: `python tests/test_basic.py` - Start here

## Documentation by File Location

### Root Directory
- README.md - Quick start and overview
- PROJECT_SUMMARY.md - Problem statement and vision
- DEEP_LEARNING_GUIDE.md - GPU setup guide
- CHANGE_HISTORY.md - Project evolution
- cli.py - Command-line interface
- patient_cli.py - Web interface

### docs/ Directory

#### Core Documentation
- docs/DOCUMENTATION_INDEX.md ⭐ **NEW** - This file
- docs/TESTING.md ⭐ **NEW** - Comprehensive testing guide (26 tests)
- docs/TESTING_MULTI_TURN_DIALOGUES.md - Dialogue integration tests

#### Module Documentation (docs/modules/) ⭐ **NEW**
- docs/modules/DATA_MODULE.md - Data layer documentation
- docs/modules/MODELS_MODULE.md - Neural networks documentation
- docs/modules/CONVERSATION_MODULE.md - Dialogue engine documentation
- docs/modules/TRIAGE_MODULE.md - Diagnosis orchestration documentation
- docs/modules/IMPLEMENTATION_SUMMARY.md - High-level architecture
- docs/modules/IMPLEMENTATION_DETAILS.md - Deep-learning details

#### Guides (docs/guides/) ⭐ **NEW**
- docs/guides/SOP.md - Complete Standard Operating Procedure
- docs/guides/PHYSICIAN_CONFIGURATION_GUIDE.md - Configuration guide for physicians

#### Updates (docs/updates/) ⭐ **NEW**
- docs/updates/UPDATE_LOG.md - Consolidated update history
- docs/updates/REORGANIZATION_COMPLETE.md - Repository reorganization summary
- docs/updates/REORGANIZATION_SUMMARY.md - Detailed reorganization changes

#### Architecture Documentation (docs/architecture/)
- docs/architecture/DIALOGUE_ENGINE.md
- docs/architecture/IMPLEMENTATION_DIALOGUE_ENGINE.md
- docs/architecture/DIAGNOSIS_ORCHESTRATOR_README.md

#### Feature Documentation (docs/features/)
- docs/features/MODEL_LOADER_GUIDE.md
- docs/features/GNN_OPTIMIZATION_GUIDE.md
- docs/features/CAUSAL_EDGES_GUIDE.md
- docs/features/TEMPORAL_MODELING_README.md
- docs/features/COMORBIDITY_IMPLEMENTATION.md
- docs/features/IMPLEMENTATION_CHECKLIST.md
- docs/features/TEMPLATE_IMPLEMENTATION.md
- docs/features/TEMPLATE_QUICKSTART.md

### tests/ Directory ⭐ **REORGANIZED**
All test files (26 total) now in `tests/` directory:
- tests/test_basic.py - Core tests
- tests/test_enhanced_bayesian.py - Bayesian features
- tests/test_dialogue_engine.py - Dialogue tests
- tests/test_diagnosis_orchestrator.py - Diagnosis tests
- tests/test_conversation_flow.py - Integration tests
- ... and 21 more (see [TESTING.md](TESTING.md))

### demos/ Directory ⭐ **NEW**
All demo files (10 total) now in `demos/` directory:
- demos/simple_demo.py - Quick demo
- demos/demo_dialogue_engine.py - Dialogue demo
- demos/demo_deep_learning.py - Full system demo
- ... and 7 more

### config/ Directory
- config/respiratory_conditions.yaml - Medical conditions
- config/red_flags.yaml - Emergency criteria
- config/comorbidity_effects.yaml - Comorbidity modeling
- config/symptom_causality.yaml - Causal relationships

### .github/ Directory
- .github/copilot-instructions.md - Instructions for GitHub Copilot

## Document Status

### Core Documentation
- ✅ README.md - **UPDATED** with new structure
- ✅ PROJECT_SUMMARY.md - Current
- ✅ CHANGE_HISTORY.md - Current
- ✅ docs/DOCUMENTATION_INDEX.md - **UPDATED** with new paths

### Module Documentation ⭐ **NEW**
- ✅ docs/modules/DATA_MODULE.md - **NEW** comprehensive data layer guide
- ✅ docs/modules/MODELS_MODULE.md - **NEW** comprehensive models guide
- ✅ docs/modules/CONVERSATION_MODULE.md - **NEW** comprehensive dialogue guide
- ✅ docs/modules/TRIAGE_MODULE.md - **NEW** comprehensive triage guide
- ✅ docs/modules/IMPLEMENTATION_SUMMARY.md - **MOVED** from root
- ✅ docs/modules/IMPLEMENTATION_DETAILS.md - **MOVED** from root

### Guides ⭐ **NEW**
- ✅ docs/guides/SOP.md - **NEW** comprehensive Standard Operating Procedure
- ✅ docs/guides/PHYSICIAN_CONFIGURATION_GUIDE.md - Configuration guide for physicians

### Updates ⭐ **NEW**
- ✅ docs/updates/UPDATE_LOG.md - **NEW** consolidated update history
- ✅ docs/updates/REORGANIZATION_COMPLETE.md - **MOVED** from root
- ✅ docs/updates/REORGANIZATION_SUMMARY.md - Detailed reorganization changes
- 🗑️ NORMALIZATION_FIX_SUMMARY.md - **REMOVED** (consolidated)
- 🗑️ VERIFICATION_REPORT.md - **REMOVED** (consolidated)

### Testing Documentation
- ✅ docs/TESTING.md - **UPDATED** for 26 test files
- ✅ docs/TESTING_MULTI_TURN_DIALOGUES.md - Current
- ✅ .github/copilot-instructions.md - **UPDATED** with correct paths

### Feature Documentation
- ✅ All feature docs current and accurate

### Repository Organization ⭐ **REORGANIZED**
- ✅ All tests moved to `tests/` directory (26 files)
- ✅ All demos moved to `demos/` directory (10 files)
- ✅ Test imports updated for new location
- ✅ All tests verified working from new location

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
