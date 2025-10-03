#!/usr/bin/env python3
"""
Demonstration of comorbidity modeling in the Enhanced Bayesian Network.

This demo shows how comorbidities affect symptom presentations, including:
1. Single comorbidity effects on symptom probabilities
2. Multiple comorbidities compounding effects
3. Comorbidity-specific symptoms
4. Cross-condition interactions (e.g., Asthma + COPD = ACOS)
"""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from phaita.models.enhanced_bayesian_network import create_enhanced_bayesian_network


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_case(case_num, description, symptoms, metadata):
    """Print a formatted case result."""
    print(f"\n{case_num}. {description}")
    print(f"   Symptoms ({len(symptoms)}): {', '.join(symptoms)}")
    print(f"   Metadata: {metadata}")


def main():
    """Run comorbidity modeling demonstrations."""
    print("🏥 PHAITA Comorbidity Modeling Demonstration")
    print("=" * 70)
    
    network = create_enhanced_bayesian_network()
    
    # Demo 1: Single comorbidity effects
    print_section("Demo 1: Single Comorbidity Effects")
    
    print("\n📋 Baseline: Asthma without comorbidities")
    symptoms_baseline, metadata_baseline = network.sample_symptoms("J45.9")
    print_case("1a", "Asthma (no comorbidities)", symptoms_baseline, metadata_baseline)
    
    print("\n📋 With diabetes comorbidity")
    symptoms_diabetes, metadata_diabetes = network.sample_symptoms(
        "J45.9", 
        comorbidities=["diabetes"]
    )
    print_case("1b", "Asthma + Diabetes", symptoms_diabetes, metadata_diabetes)
    print("   Note: Diabetes increases fatigue and infection risk probability")
    
    print("\n📋 With obesity comorbidity")
    symptoms_obesity, metadata_obesity = network.sample_symptoms(
        "J45.9", 
        comorbidities=["obesity"]
    )
    print_case("1c", "Asthma + Obesity", symptoms_obesity, metadata_obesity)
    print("   Note: Obesity increases shortness of breath and exercise intolerance")
    
    # Demo 2: Multiple comorbidities
    print_section("Demo 2: Multiple Comorbidities (Compounding Effects)")
    
    symptoms_multi, metadata_multi = network.sample_symptoms(
        "J45.9",
        comorbidities=["diabetes", "obesity", "hypertension"]
    )
    print_case("2a", "Asthma + Multiple Comorbidities", symptoms_multi, metadata_multi)
    print("   Note: Effects are multiplicative - probabilities compound")
    print("   - Diabetes: ↑ fatigue, infection risk")
    print("   - Obesity: ↑ dyspnea, exercise intolerance")
    print("   - Hypertension: ↑ dyspnea, chest pain")
    
    # Demo 3: Comorbidity-specific symptoms
    print_section("Demo 3: Comorbidity-Specific Symptoms")
    
    print("\n📋 Hypertension adds specific symptoms")
    symptoms_htn, metadata_htn = network.sample_symptoms(
        "J45.9",
        comorbidities=["hypertension"]
    )
    print_case("3a", "Asthma + Hypertension", symptoms_htn, metadata_htn)
    print("   Look for: palpitations, dizziness, headache (hypertension-specific)")
    
    print("\n📋 Anxiety adds specific symptoms")
    symptoms_anxiety, metadata_anxiety = network.sample_symptoms(
        "J45.9",
        comorbidities=["anxiety"]
    )
    print_case("3b", "Asthma + Anxiety", symptoms_anxiety, metadata_anxiety)
    print("   Look for: sense_of_doom, trembling, sweating (anxiety-specific)")
    
    # Demo 4: Cross-condition interactions (ACOS)
    print_section("Demo 4: Cross-Condition Interactions (ACOS)")
    
    print("\n📋 Asthma-COPD Overlap Syndrome (ACOS)")
    print("   Clinical significance: GINA/GOLD Guidelines 2023")
    print("   Key feature: chronic_cough probability increases to 90%")
    
    acos_cases = []
    for i in range(5):
        symptoms_acos, metadata_acos = network.sample_symptoms(
            "J45.9",
            comorbidities=["copd"]
        )
        has_chronic = "chronic_cough" in symptoms_acos
        acos_cases.append(has_chronic)
        marker = "✓" if has_chronic else "✗"
        print(f"   {marker} Trial {i+1}: chronic_cough={'present' if has_chronic else 'absent'}")
        if i == 0:
            print_case("4a", "Asthma + COPD (ACOS)", symptoms_acos, metadata_acos)
    
    chronic_rate = sum(acos_cases) / len(acos_cases) * 100
    print(f"\n   Chronic cough rate: {chronic_rate:.0f}% (expected ~90%)")
    
    # Demo 5: Different severities with comorbidities
    print_section("Demo 5: Severity + Comorbidity Interactions")
    
    print("\n📋 Mild severity with comorbidity")
    symptoms_mild, metadata_mild = network.sample_symptoms(
        "J45.9",
        comorbidities=["obesity"],
        severity="mild"
    )
    print_case("5a", "Mild Asthma + Obesity", symptoms_mild, metadata_mild)
    
    print("\n📋 Severe severity with comorbidity")
    symptoms_severe, metadata_severe = network.sample_symptoms(
        "J45.9",
        comorbidities=["obesity"],
        severity="severe"
    )
    print_case("5b", "Severe Asthma + Obesity", symptoms_severe, metadata_severe)
    print(f"   Note: Severe case has {len(symptoms_severe)} symptoms vs {len(symptoms_mild)} in mild case")
    
    # Demo 6: Pneumonia with immunocompromised comorbidity
    print_section("Demo 6: Pneumonia + Immunocompromised")
    
    symptoms_immuno, metadata_immuno = network.sample_symptoms(
        "J18.9",  # Pneumonia
        comorbidities=["immunocompromised"]
    )
    print_case("6a", "Pneumonia + Immunocompromised", symptoms_immuno, metadata_immuno)
    print("   Note: Increased infection risk (1.8x), fever (1.4x)")
    print("   Look for: recurrent_infections, prolonged_illness, night_sweats")
    
    # Summary
    print_section("Summary")
    print("""
    The Enhanced Bayesian Network with comorbidity modeling provides:
    
    ✓ Evidence-based symptom probability modifiers
    ✓ Comorbidity-specific symptom additions
    ✓ Cross-condition interaction effects (e.g., ACOS)
    ✓ Multiplicative effects for multiple comorbidities
    ✓ Priority preservation of high-probability interaction symptoms
    ✓ Clinical references in config/comorbidity_effects.yaml
    
    Usage:
        symptoms, metadata = network.sample_symptoms(
            condition_code="J45.9",
            comorbidities=["diabetes", "obesity"],
            age_group="adult",
            severity="moderate"
        )
    
    For more details, see:
    - config/comorbidity_effects.yaml (clinical evidence references)
    - test_enhanced_bayesian.py (comprehensive test cases)
    - phaita/models/enhanced_bayesian_network.py (implementation)
    """)
    
    print("\n" + "=" * 70)
    print("🎉 Comorbidity modeling demonstration complete!")


if __name__ == "__main__":
    main()
