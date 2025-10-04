# Data Module Documentation

**Location:** `phaita/data/`

## Overview

The data module handles all medical condition data, forum scraping for realistic complaints, synthetic data generation, text preprocessing, and red-flag definitions.

## Components

### 1. ICD Conditions (`icd_conditions.py`)

**Purpose:** Central repository for ICD-10 respiratory condition definitions.

**Key Class:** `RespiratoryConditions`

**Features:**
- 10 ICD-10 respiratory conditions
- Structured symptom profiles per condition
- Condition metadata (code, name, description)
- Symptom probability mappings

**Usage:**
```python
from phaita.data.icd_conditions import RespiratoryConditions

# Get all conditions
conditions = RespiratoryConditions.get_all_conditions()

# Get specific condition
asthma = RespiratoryConditions.get_condition("J45.9")

# Get symptoms for a condition
symptoms = RespiratoryConditions.get_symptoms("J45.9")
```

**Conditions Included:**
1. J45.9 - Asthma, unspecified
2. J44.0 - COPD with acute lower respiratory infection
3. J18.9 - Pneumonia, unspecified organism
4. J06.9 - Acute upper respiratory infection, unspecified
5. J20.9 - Acute bronchitis, unspecified
6. J42 - Chronic bronchitis, unspecified
7. J84.9 - Interstitial pulmonary disease, unspecified
8. J81.0 - Acute pulmonary edema
9. J96.0 - Acute respiratory failure
10. Multiple - Asthma-COPD overlap syndrome (ACOS)

---

### 2. Forum Scraper (`forum_scraper.py`)

**Purpose:** Scrape and process patient complaints from online forums for realistic training data.

**Key Classes:**
- `ForumScraper` - Main scraping interface
- `RedditScraper` - Reddit-specific implementation

**Features:**
- Scrape medical subreddits (r/AskDocs, r/medical)
- Extract lay language symptom descriptions
- Map lay terms to medical terminology
- Privacy-preserving data collection

**Usage:**
```python
from phaita.data.forum_scraper import ForumScraper

scraper = ForumScraper()
complaints = scraper.scrape_subreddit("AskDocs", limit=100)
```

**Privacy Notes:**
- Removes personally identifiable information
- Anonymizes user data
- Complies with platform terms of service

---

### 3. Synthetic Generator (`synthetic_generator.py`)

**Purpose:** Generate synthetic patient complaints for training and evaluation.

**Key Class:** `SyntheticGenerator`

**Features:**
- Batch generation of realistic complaints
- Condition-aware generation
- Demographic variation
- Severity levels (mild, moderate, severe)
- Age-appropriate symptom selection

**Usage:**
```python
from phaita.data.synthetic_generator import SyntheticGenerator

generator = SyntheticGenerator()

# Generate single complaint
complaint = generator.generate_complaint(
    condition_code="J45.9",
    severity="moderate",
    age=45
)

# Batch generation
batch = generator.generate_batch(
    count=100,
    conditions=["J45.9", "J44.0", "J18.9"]
)
```

**Generation Parameters:**
- `condition_code`: Target ICD-10 code
- `severity`: "mild", "moderate", "severe"
- `age`: Patient age (affects symptom selection)
- `comorbidities`: List of comorbid conditions
- `rare`: Generate rare presentation (bool)

---

### 4. Preprocessing (`preprocessing.py`)

**Purpose:** Text preprocessing and normalization utilities.

**Key Functions:**
- `normalize_text()` - Clean and normalize text
- `tokenize_medical_text()` - Medical-aware tokenization
- `extract_symptoms()` - Extract symptom mentions from text

**Features:**
- Medical abbreviation expansion
- Spell correction
- Symptom normalization
- Negation detection

**Usage:**
```python
from phaita.data.preprocessing import normalize_text, extract_symptoms

# Normalize text
clean_text = normalize_text("Pt c/o SOB x 3 days")
# Result: "Patient complains of shortness of breath for 3 days"

# Extract symptoms
symptoms = extract_symptoms("I have chest pain and shortness of breath")
# Result: ["chest_pain", "shortness_of_breath"]
```

**Normalization Rules:**
- Lowercase conversion
- Underscore/hyphen to space
- Medical abbreviation expansion
- Whitespace trimming

---

### 5. Red Flags (`red_flags.py`)

**Purpose:** Load and provide access to red-flag symptoms that require immediate medical attention.

**Configuration:** Red flag definitions are stored in `config/red_flags.yaml` for easy clinician updates.

**Key Export:** `RESPIRATORY_RED_FLAGS`

**Features:**
- Condition-specific red flags (technical names for matching)
- Human-readable symptom descriptions (for display)
- Emergency escalation guidance text
- Loaded from YAML configuration at import time

**Usage:**
```python
from phaita.data.red_flags import RESPIRATORY_RED_FLAGS

# Get red flags for condition
condition_data = RESPIRATORY_RED_FLAGS.get("J45.9", {})

# Access technical red-flag names (for matching)
red_flags = condition_data.get("red_flags", [])
# ['severe_respiratory_distress', 'unable_to_speak_full_sentences', ...]

# Access human-readable symptoms (for display)
symptoms = condition_data.get("symptoms", [])
# ['inability to speak more than a few words', 'bluish lips or fingernails', ...]

# Get escalation guidance
escalation_text = condition_data.get("escalation", "")
# "Use a rescue inhaler immediately and seek emergency care..."
```

**YAML Configuration Structure:**
```yaml
J45.9:  # Asthma
  red_flags:
    - severe_respiratory_distress
    - unable_to_speak_full_sentences
  symptoms:
    - inability to speak more than a few words
    - bluish lips or fingernails
  escalation: Use a rescue inhaler immediately...
```

**Clinician Updates:**
Clinicians can update red flag definitions by editing `config/red_flags.yaml` without modifying Python code. Changes take effect on next import/restart.

---

### 6. Template Loader (`template_loader.py`)

**Purpose:** Load and manage complaint generation templates.

**Key Class:** `TemplateLoader`

**Features:**
- Template loading from YAML files
- Variable substitution
- Grammar-aware generation
- Fallback templates for CPU-only mode

**Usage:**
```python
from phaita.data.template_loader import TemplateLoader

loader = TemplateLoader()
templates = loader.load_templates("config/templates.yaml")
```

**Template Format:**
```yaml
templates:
  - pattern: "I've been experiencing {symptom} for {duration}"
    variables:
      symptom: ["shortness of breath", "chest pain", "wheezing"]
      duration: ["a few hours", "2 days", "several weeks"]
```

---

## Configuration Files

The data module uses several configuration files in `config/`:

### `respiratory_conditions.yaml`
Detailed condition definitions with symptom probabilities.

### `red_flags.yaml`
Red-flag definitions with severity and guidance.

### `symptom_causality.yaml`
Causal relationships between symptoms.

### `comorbidity_effects.yaml`
How comorbidities affect symptom presentation.

---

## Testing

**Test Files:**
- `tests/test_basic.py` - Core data layer tests
- `tests/test_forum_scraping.py` - Forum scraper tests
- `tests/test_conditions_config.py` - Configuration tests

**Run Tests:**
```bash
python tests/test_basic.py
python tests/test_forum_scraping.py
```

---

## Integration Examples

### Example 1: Generate Training Data

```python
from phaita.data.icd_conditions import RespiratoryConditions
from phaita.data.synthetic_generator import SyntheticGenerator

# Initialize
conditions = RespiratoryConditions.get_all_conditions()
generator = SyntheticGenerator()

# Generate diverse training set
training_data = []
for condition in conditions:
    for severity in ["mild", "moderate", "severe"]:
        complaints = generator.generate_batch(
            count=10,
            conditions=[condition["code"]],
            severity=severity
        )
        training_data.extend(complaints)

print(f"Generated {len(training_data)} training examples")
```

### Example 2: Check for Red Flags

```python
from phaita.data.red_flags import RESPIRATORY_RED_FLAGS

# Patient complaint with detected symptoms
detected_symptoms = ["severe_respiratory_distress", "wheezing", "cough"]

# Check for red flags for asthma (J45.9)
condition_code = "J45.9"
condition_data = RESPIRATORY_RED_FLAGS.get(condition_code, {})

# Get red-flag definitions (technical names)
red_flag_list = condition_data.get("red_flags", [])

# Check which red flags are present
present_red_flags = [rf for rf in red_flag_list if rf in detected_symptoms]

if present_red_flags:
    print("⚠️ RED FLAGS DETECTED:")
    # Get human-readable symptoms
    symptoms = condition_data.get("symptoms", [])
    print(f"  Symptoms: {', '.join(symptoms)}")
    
    # Get escalation guidance
    escalation = condition_data.get("escalation", "")
    print(f"  🚑 Escalation: {escalation}")
```

---

## Best Practices

### DO:
- ✅ Use `RespiratoryConditions` for all condition data
- ✅ Normalize symptoms before matching
- ✅ Check red flags on every patient input
- ✅ Generate diverse training data with various severities
- ✅ Edit `config/red_flags.yaml` to update red flag definitions (clinician-friendly)

### DON'T:
- ❌ Hardcode condition or symptom data
- ❌ Skip normalization - causes matching failures
- ❌ Ignore red flags - patient safety critical
- ❌ Use PII from forum data without anonymization
- ❌ Edit `phaita/data/red_flags.py` - use YAML config instead

---

## Related Documentation

- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Overall architecture
- [Bayesian Module](BAYESIAN_MODULE.md) - How conditions connect to probabilities
- [Triage Module](TRIAGE_MODULE.md) - How red flags integrate into triage
- [UPDATE_LOG.md](../updates/UPDATE_LOG.md) - Recent data module updates

---

## API Reference

See inline docstrings in each module for detailed API documentation:
- `phaita/data/icd_conditions.py`
- `phaita/data/forum_scraper.py`
- `phaita/data/synthetic_generator.py`
- `phaita/data/preprocessing.py`
- `phaita/data/red_flags.py`
- `phaita/data/template_loader.py`

---

**Last Updated:** 2025-01-03
