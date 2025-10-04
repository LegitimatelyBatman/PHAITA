# PHAITA Physician Configuration Guide
## A Step-by-Step Guide for Clinicians with Minimal Coding Experience

---

## 📋 **Table of Contents**

1. [Introduction](#introduction)
2. [Before You Begin](#before-you-begin)
3. [Understanding YAML Format](#understanding-yaml-format)
4. [File 1: Respiratory Conditions](#file-1-respiratory-conditions)
5. [File 2: Red-Flag Symptoms](#file-2-red-flag-symptoms)
6. [File 3: Comorbidity Effects](#file-3-comorbidity-effects)
7. [File 4: Symptom Causality](#file-4-symptom-causality)
8. [Testing Your Changes](#testing-your-changes)
9. [Common Mistakes to Avoid](#common-mistakes-to-avoid)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

Welcome! This guide will help you customize the clinical knowledge in PHAITA without needing programming experience. You'll learn to edit four important configuration files that control how the system understands medical conditions, symptoms, and patient presentations.

**What you can do:**
- Add new medical conditions
- Update symptom lists
- Define emergency red-flags
- Model how comorbidities affect symptom presentation
- Set up causal relationships between symptoms

**What you don't need:**
- Programming knowledge
- Command line expertise
- Understanding of neural networks

---

## Before You Begin

### ✅ **Required Tools**

1. **Text Editor** (choose ONE):
   - **Recommended:** [Visual Studio Code](https://code.visualstudio.com/) (free, user-friendly)
   - Alternative: [Notepad++](https://notepad-plus-plus.org/) (Windows)
   - Alternative: [Sublime Text](https://www.sublimetext.com/)
   - ⚠️ **DO NOT USE:** Microsoft Word, Google Docs (they add formatting that breaks the files)

2. **The PHAITA Project Folder:**
   - Location: Where you installed PHAITA (example: `C:\PHAITA\` or `/home/username/PHAITA/`)
   - Inside you'll find a `config/` folder - this is where we'll work

---

## Understanding YAML Format

The configuration files use YAML format - it's like organized bullet points for computers.

### **Basic Rules:**

1. **Indentation matters** - Use SPACES, not tabs
2. **Colons** separate names from values: `symptom: cough`
3. **Dashes** create lists:
   ```yaml
   symptoms:
     - cough
     - fever
     - fatigue
   ```
4. **Comments** start with `#` and are ignored by the computer:
   ```yaml
   # This is a comment - add notes for yourself!
   fever: 38.5  # Temperature in Celsius
   ```

### **Visual Guide:**
```yaml
condition_name:              # Main category (no indent)
  subcategory:               # Subcategory (2 spaces indent)
    - item_one               # List item (4 spaces + dash)
    - item_two               # Same indentation
  another_subcategory:       # Back to 2 spaces
    value: 1.5               # Key-value pair
```

---

## File 1: Respiratory Conditions

**Location:** `config/respiratory_conditions.yaml`

This file defines medical conditions, their symptoms, and patient demographics.

### **Structure Overview:**

```yaml
J45.9:                        # ICD-10 code (like a medical ID)
  name: Asthma                # Condition name
  symptoms:                   # List of clinical symptoms
    - wheezing
    - shortness_of_breath
  severity_indicators:        # Emergency signs
    - unable_to_speak
  lay_terms:                  # How patients describe it
    - "can't breathe"
    - wheezy
  demographics:               # Who gets this condition
    inclusion:
      age_ranges:
        - min: 12
          max: 45
```

### **Step-by-Step: Adding a New Condition**

#### **Step 1: Find the right location**
1. Open `config/respiratory_conditions.yaml` in your text editor
2. Scroll to the bottom of the file
3. Add a blank line after the last condition

#### **Step 2: Start with the ICD-10 code**
```yaml
J44.1:                        # Replace with actual ICD-10 code
```
**⚠️ Important:** Make sure the code matches standard ICD-10 format

#### **Step 3: Add the condition name**
```yaml
J44.1:
  name: Chronic Obstructive Pulmonary Disease with Acute Exacerbation
```

#### **Step 4: List clinical symptoms**
Use medical terminology - underscores connect words:
```yaml
J44.1:
  name: Chronic Obstructive Pulmonary Disease with Acute Exacerbation
  symptoms:
    - chronic_cough
    - sputum_production
    - dyspnea
    - wheezing
    - chest_tightness
```

**💡 Tip:** Use underscores `_` instead of spaces: `chest_pain` not `chest pain`

#### **Step 5: Add severity indicators**
These are signs that indicate severe disease:
```yaml
  severity_indicators:
    - severe_breathlessness
    - cyanosis
    - confusion
    - respiratory_failure
```

#### **Step 6: Include patient language (lay terms)**
How patients actually describe symptoms (use quotes for phrases with spaces):
```yaml
  lay_terms:
    - "can't catch my breath"
    - "bringing up phlegm"
    - "chest feels tight"
    - breathless
```

#### **Step 7: Add demographics**
Who typically gets this condition:
```yaml
  demographics:
    inclusion:
      age_ranges:
        - min: 55              # Minimum age
          max: 85              # Maximum age
          weight: 1.0          # How common (1.0 = very common)
      sexes:
        - male
        - female
      social_history:
        - Former smoker
        - Occupational dust exposure
```

#### **Step 8: Add medical history context**
```yaml
  history:
    inclusion:
      past_conditions:
        - Chronic bronchitis
        - Emphysema
      medications:
        - Long-acting bronchodilator
      recent_events:
        - Recent respiratory infection
```

### **Complete Example:**

```yaml
J44.1:
  name: COPD with Acute Exacerbation
  symptoms:
    - chronic_cough
    - increased_sputum
    - dyspnea
    - wheezing
  severity_indicators:
    - severe_breathlessness
    - cyanosis
    - confusion
  lay_terms:
    - "can't breathe"
    - "coughing up more mucus"
    - wheezy
  description: Acute worsening of COPD symptoms requiring medical intervention
  demographics:
    inclusion:
      age_ranges:
        - min: 55
          max: 85
      sexes:
        - male
        - female
      social_history:
        - Smoking history
    exclusion:
      age_ranges:
        - min: 0
          max: 25
      notes:
        - Primary diagnosis of cystic fibrosis
  history:
    inclusion:
      past_conditions:
        - Chronic bronchitis
      medications:
        - Bronchodilator therapy
      recent_events:
        - Recent viral infection
```

---

## File 2: Red-Flag Symptoms

**Location:** `config/red_flags.yaml`

This file defines emergency symptoms that require immediate medical attention.

### **Purpose:**
Helps the system identify when patients need emergency care vs. routine appointments.

### **Structure:**

```yaml
J45.9:                        # ICD-10 code (must match conditions file)
  red_flags:                  # List of emergency symptoms
    - severe_respiratory_distress
    - unable_to_speak_full_sentences
    - cyanosis
```

### **Step-by-Step: Adding Red-Flags**

#### **Step 1: Match the condition code**
```yaml
J44.1:                        # Same code as in respiratory_conditions.yaml
  red_flags:
```

#### **Step 2: List emergency symptoms**
Think: "What symptoms mean this patient needs the ER NOW?"
```yaml
J44.1:
  red_flags:
    - severe_breathlessness_at_rest
    - altered_mental_status
    - cyanosis
    - respiratory_failure
    - severe_hypoxemia
```

### **Clinical Guidelines for Red-Flags:**

**Respiratory Emergency Indicators:**
- Severe respiratory distress
- Inability to speak in full sentences
- Oxygen saturation < 92%
- Altered mental status/confusion
- Cyanosis
- Severe chest pain
- Hemoptysis
- Rapid deterioration

### **Example:**

```yaml
J44.1:
  red_flags:
    - severe_breathlessness_at_rest
    - new_confusion
    - drowsiness
    - cyanosis
    - oxygen_saturation_below_90
    - respiratory_rate_over_30
```

**💡 Pro Tip:** When in doubt, include it as a red-flag. Better to be cautious with patient safety.

---

## File 3: Comorbidity Effects

**Location:** `config/comorbidity_effects.yaml`

This file models how chronic conditions affect symptom presentation.

### **Purpose:**
Helps the system understand that a diabetic patient with pneumonia may present differently than a non-diabetic patient.

### **Structure:**

```yaml
diabetes:                     # Comorbidity name
  symptom_modifiers:          # How it changes symptom probability
    fever: 1.3                # Multiplier (>1 = more likely)
    fatigue: 1.5
  specific_symptoms:          # Symptoms unique to this comorbidity
    - frequent_urination
  probability: 0.3            # Chance of adding specific symptoms
```

### **Understanding Multipliers:**

- `1.0` = No change
- `1.3` = 30% more likely
- `1.5` = 50% more likely
- `0.8` = 20% less likely

### **Step-by-Step: Adding a Comorbidity**

#### **Step 1: Name the comorbidity**
```yaml
chronic_kidney_disease:       # Use underscores, lowercase
```

#### **Step 2: List symptom modifiers**
Based on clinical evidence - how does this condition affect respiratory symptoms?
```yaml
chronic_kidney_disease:
  symptom_modifiers:
    fatigue: 1.4              # CKD causes fatigue
    dyspnea: 1.3              # Fluid overload
    cough: 1.2                # Pulmonary edema risk
```

#### **Step 3: Add comorbidity-specific symptoms**
```yaml
  specific_symptoms:
    - fluid_overload
    - peripheral_edema
    - reduced_exercise_tolerance
```

#### **Step 4: Set probability**
```yaml
  probability: 0.35           # 35% chance to add specific symptoms
```

#### **Step 5: Add clinical reference**
Always include evidence as a comment:
```yaml
chronic_kidney_disease:
  # Source: KDIGO CKD Guidelines 2024
  symptom_modifiers:
    fatigue: 1.4
    dyspnea: 1.3
  specific_symptoms:
    - fluid_overload
  probability: 0.35
```

### **Advanced: Cross-Condition Interactions**

Some conditions interact (like Asthma + COPD = ACOS):

```yaml
interactions:
  asthma_copd:                # Interaction name
    conditions:
      - J45.9                 # Asthma
      - J44.9                 # COPD
    symptom_modifiers:
      chronic_cough: 0.9      # 90% probability (absolute)
      wheezing: 0.88
    note: "Asthma-COPD Overlap Syndrome (ACOS)"
```

---

## File 4: Symptom Causality

**Location:** `config/symptom_causality.yaml`

This file defines how symptoms cause or lead to other symptoms.

### **Purpose:**
Models symptom progression - for example, severe cough can cause chest pain.

### **Three Types of Relationships:**

1. **Co-occurrence:** Symptoms that happen together
2. **Causal:** One symptom causes another
3. **Temporal:** One symptom precedes another

### **Structure:**

```yaml
causal_edges:
  - from: severe_cough       # Cause symptom
    to: chest_pain           # Effect symptom
    weight: 0.75             # Strength (0-1)
    type: causal             # Relationship type
```

### **Step-by-Step: Adding a Causal Link**

#### **Step 1: Identify the clinical relationship**
Ask: "Does symptom A directly cause symptom B?"

Example: Severe dyspnea → Anxiety

#### **Step 2: Add the edge**
```yaml
causal_edges:
  - from: severe_dyspnea
    to: anxiety
    weight: 0.65
    type: causal
    clinical_rationale: "Dyspnea commonly triggers anxiety response"
```

#### **Step 3: Set the weight**
Weight = How strong is the relationship?
- `0.9` = Very strong (almost always)
- `0.7` = Strong (usually)
- `0.5` = Moderate (sometimes)
- `0.3` = Weak (occasionally)

### **Temporal Relationships:**

When one symptom typically comes before another:

```yaml
temporal_edges:
  - from: nasal_congestion
    to: post_nasal_drip
    weight: 0.7
    delay: 1.0                # Days delay
    type: temporal
    clinical_rationale: "Congestion leads to PND within 24-48 hours"
```

### **Complete Example:**

```yaml
causal_edges:
  - from: severe_cough
    to: chest_pain
    weight: 0.75
    type: causal
    clinical_rationale: "Persistent coughing causes musculoskeletal chest pain"
  
  - from: productive_cough
    to: fatigue
    weight: 0.65
    type: causal
    clinical_rationale: "Sleep disruption from cough leads to fatigue"

temporal_edges:
  - from: fever
    to: sweating
    weight: 0.85
    delay: 0.5                # Hours
    type: temporal
    clinical_rationale: "Fever typically followed by sweating as temperature resolves"
```

---

## Testing Your Changes

After editing any configuration file, you MUST test your changes.

### **Step 1: Save the file**
- File → Save (or Ctrl+S / Cmd+S)
- Make sure it's saved in the correct location

### **Step 2: Check YAML syntax**
Open a terminal/command prompt and run:

```bash
# Navigate to PHAITA folder
cd /path/to/PHAITA

# Check the file syntax
python -c "import yaml; yaml.safe_load(open('config/respiratory_conditions.yaml'))"
```

**If successful:** No output = good!  
**If error:** You'll see which line has the problem

### **Step 3: Run basic tests**

```bash
# Test that conditions load correctly
python tests/test_basic.py

# Test red-flags system
python tests/test_diagnosis_orchestrator.py

# Test comorbidities
python tests/test_enhanced_bayesian.py
```

**Look for:** ✅ All tests passed

### **Step 4: Try a demo**

```bash
# See your changes in action
python demos/simple_demo.py
```

---

## Common Mistakes to Avoid

### ❌ **Mistake 1: Using Tabs Instead of Spaces**
```yaml
# WRONG:
J45.9:
→symptoms:              # Tab used (invisible!)

# CORRECT:
J45.9:
  symptoms:             # 2 spaces
```

### ❌ **Mistake 2: Inconsistent Indentation**
```yaml
# WRONG:
symptoms:
  - cough
   - fever              # 3 spaces instead of 2

# CORRECT:
symptoms:
  - cough
  - fever               # Consistent 2 spaces
```

### ❌ **Mistake 3: Forgetting Colons**
```yaml
# WRONG:
name Asthma             # Missing colon

# CORRECT:
name: Asthma            # Colon after key
```

### ❌ **Mistake 4: Spaces in Symptom Names**
```yaml
# WRONG:
symptoms:
  - chest pain          # Space breaks the system

# CORRECT:
symptoms:
  - chest_pain          # Underscore connects words
```

### ❌ **Mistake 5: Wrong ICD-10 Code Format**
```yaml
# WRONG:
J459:                   # Missing decimal
45.9:                   # Missing letter

# CORRECT:
J45.9:                  # Letter + numbers + decimal + number
```

### ❌ **Mistake 6: Mismatched Codes**
```yaml
# In respiratory_conditions.yaml:
J45.9:
  name: Asthma

# In red_flags.yaml:
J45.8:                  # WRONG - doesn't match!
  red_flags:
    - severe_distress
```

**Rule:** ICD codes must match EXACTLY across all files

---

## Troubleshooting

### **Problem: "YAML parse error"**

**Symptoms:**
```
yaml.scanner.ScannerError: while scanning for the next token
found character '\t' that cannot start any token
```

**Solution:**
1. Check for tabs - replace with spaces
2. Check indentation is consistent (2 spaces per level)
3. Make sure colons have spaces after them: `name: value` not `name:value`

---

### **Problem: "Symptom not recognized"**

**Symptoms:**
- Red-flags not triggering
- Symptoms not appearing in output

**Solution:**
1. Check spelling is identical in all files
2. Use underscores, not spaces: `chest_pain` not `chest pain`
3. Make sure symptom is listed in the condition's symptoms list

---

### **Problem: Tests fail after changes**

**Symptoms:**
```
❌ test_basic.py failed: Condition J99.9 not found
```

**Solution:**
1. Make sure ICD-10 code exists in `respiratory_conditions.yaml`
2. Check all required fields are present: name, symptoms, description
3. Verify YAML formatting is correct

---

### **Problem: Changes not appearing in system**

**Symptoms:**
- Made changes but system uses old data

**Solution:**
1. Make sure you saved the file (check the file modification date)
2. Restart the PHAITA application
3. Check you edited the correct file in the correct location

---

### **Getting Help:**

1. **Check the file examples** - Look at existing conditions for reference
2. **Use YAML validator** - https://www.yamllint.com/ (paste your code)
3. **Review test output** - Error messages often point to the exact problem
4. **Ask for help** - Include the error message and which file you edited

---

## Quick Reference Card

### **File Locations:**
```
PHAITA/
└── config/
    ├── respiratory_conditions.yaml    # Conditions & symptoms
    ├── red_flags.yaml                 # Emergency symptoms
    ├── comorbidity_effects.yaml       # Comorbidity effects
    └── symptom_causality.yaml         # Symptom relationships
```

### **YAML Syntax Cheat Sheet:**
```yaml
# Comment (ignored by computer)
key: value                    # Key-value pair
list:                         # List
  - item1                     # Dash + space + item
  - item2
nested:                       # Nested structure
  subkey: value              # Indent with 2 spaces
  another:
    deep: value              # Indent again (4 spaces total)
```

### **Test Commands:**
```bash
# Basic test
python tests/test_basic.py

# Check red-flags
python tests/test_diagnosis_orchestrator.py

# Validate YAML
python -c "import yaml; yaml.safe_load(open('config/YOUR_FILE.yaml'))"
```

### **Symptom Naming Rules:**
- ✅ Use: `chest_pain`, `shortness_of_breath`, `severe_cough`
- ❌ Avoid: `chest pain`, `shortness-of-breath`, `Severe Cough`

### **Weight/Probability Guide:**
- `0.3` = Occasionally (30%)
- `0.5` = Sometimes (50%)
- `0.7` = Usually (70%)
- `0.9` = Almost always (90%)
- `1.5` = 50% increase in likelihood
- `2.0` = Double the likelihood

---

## Appendix: Template Library

### **Template: New Respiratory Condition**

```yaml
J99.9:
  name: [Condition Name]
  symptoms:
    - [symptom_1]
    - [symptom_2]
    - [symptom_3]
  severity_indicators:
    - [severe_symptom_1]
    - [severe_symptom_2]
  lay_terms:
    - "[patient description 1]"
    - "[patient description 2]"
  description: [Clinical description of condition]
  demographics:
    inclusion:
      age_ranges:
        - min: [minimum_age]
          max: [maximum_age]
      sexes:
        - [male/female/any]
      social_history:
        - [relevant history]
  history:
    inclusion:
      past_conditions:
        - [related condition]
      medications:
        - [typical medications]
```

### **Template: Red-Flags Entry**

```yaml
J99.9:
  red_flags:
    - [emergency_symptom_1]
    - [emergency_symptom_2]
    - [emergency_symptom_3]
```

### **Template: Comorbidity**

```yaml
[comorbidity_name]:
  # Source: [Clinical guideline reference]
  symptom_modifiers:
    [symptom_1]: [multiplier]
    [symptom_2]: [multiplier]
  specific_symptoms:
    - [unique_symptom_1]
    - [unique_symptom_2]
  probability: [0.0-1.0]
```

### **Template: Causal Relationship**

```yaml
causal_edges:
  - from: [cause_symptom]
    to: [effect_symptom]
    weight: [0.0-1.0]
    type: causal
    clinical_rationale: "[explanation]"
```

---

## Summary Checklist

Before finishing your edits:

- [ ] All files saved
- [ ] YAML syntax validated (no error when checking)
- [ ] ICD-10 codes consistent across files
- [ ] Symptom names use underscores (no spaces)
- [ ] Indentation is consistent (2 spaces per level)
- [ ] Clinical references added as comments
- [ ] Tests run successfully
- [ ] Demo works with new changes
- [ ] Documented what you changed (keep notes!)

---

## Additional Resources

- **YAML Tutorial:** https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started
- **ICD-10 Lookup:** https://www.icd10data.com/
- **PHAITA Documentation:** See `docs/DOCUMENTATION_INDEX.md` in the project folder

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-03  
**For:** PHAITA Medical Triage System  
**Audience:** Physicians with minimal coding experience

---

*Remember: When in doubt, look at existing examples in the files. They're your best templates!*
