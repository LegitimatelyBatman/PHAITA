#!/usr/bin/env python3
"""
Simple demo script for PHAITA without heavy dependencies.
Shows the data layer and basic functionality.
"""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from phaita.data.icd_conditions import RespiratoryConditions
import random
import json


def demo_conditions():
    """Demonstrate respiratory conditions data."""
    print("🏥 PHAITA Demo - Respiratory Conditions")
    print("=" * 50)
    
    conditions = RespiratoryConditions.get_all_conditions()
    print(f"📋 Available conditions: {len(conditions)}\n")
    
    for i, (code, data) in enumerate(conditions.items(), 1):
        print(f"{i:2d}. {code}: {data['name']}")
        print(f"    Symptoms: {len(data['symptoms'])}, Severity indicators: {len(data['severity_indicators'])}")
        print(f"    Lay terms: {data['lay_terms'][:2]}...")
        print()


def demo_symptom_generation():
    """Demonstrate symptom generation using Bayesian network logic."""
    print("🔬 Symptom Generation Demo")
    print("=" * 40)
    
    # Simple probabilistic symptom generation
    for i in range(5):
        code, condition_data = RespiratoryConditions.get_random_condition()
        
        # Simulate symptom sampling with probabilities
        primary_symptoms = condition_data["symptoms"]
        severity_symptoms = condition_data["severity_indicators"]
        
        # Select 3-6 symptoms randomly
        num_symptoms = random.randint(3, 6)
        
        # Higher probability for primary symptoms
        selected_symptoms = []
        
        # Add 2-3 primary symptoms
        selected_symptoms.extend(random.sample(primary_symptoms, min(3, len(primary_symptoms))))
        
        # Maybe add severity indicators (30% chance each)
        for symptom in severity_symptoms:
            if random.random() < 0.3 and len(selected_symptoms) < num_symptoms:
                selected_symptoms.append(symptom)
        
        # Fill remaining with random primary symptoms
        remaining_primary = [s for s in primary_symptoms if s not in selected_symptoms]
        if len(selected_symptoms) < num_symptoms and remaining_primary:
            additional = random.sample(remaining_primary, 
                                     min(num_symptoms - len(selected_symptoms), 
                                         len(remaining_primary)))
            selected_symptoms.extend(additional)
        
        print(f"Example {i+1}: {code} - {condition_data['name']}")
        print(f"  Generated symptoms: {selected_symptoms}")
        print()


def demo_complaint_simulation():
    """Demonstrate complaint generation simulation."""
    print("💬 Patient Complaint Simulation")
    print("=" * 40)
    
    # Template-based complaint generation (simulating Mistral output)
    complaint_templates = [
        "I've been experiencing {symptoms} for the past {time}. It's really {severity}.",
        "Doctor, I have {symptoms} and I'm {feeling}. This started {time} ago.",
        "I'm having trouble with {symptoms}. It's been {severity} since {time}.",
        "Help, I can't stop {main_symptom}. I also have {other_symptoms}.",
        "I've been {main_symptom} and feeling {feeling}. It's been going on for {time}."
    ]
    
    severity_terms = ["mild", "moderate", "severe", "terrible", "awful"]
    time_terms = ["a few hours", "yesterday", "two days", "this morning", "last night"]
    feeling_terms = ["worried", "scared", "exhausted", "panicked", "terrible"]
    
    for i in range(5):
        code, condition_data = RespiratoryConditions.get_random_condition()
        lay_terms = condition_data["lay_terms"]
        
        # Select template and fill
        template = random.choice(complaint_templates)
        
        # Create complaint
        if "{symptoms}" in template:
            symptoms_text = " and ".join(random.sample(lay_terms, min(2, len(lay_terms))))
            template = template.replace("{symptoms}", symptoms_text)
        
        if "{main_symptom}" in template:
            main_symptom = random.choice(lay_terms)
            template = template.replace("{main_symptom}", main_symptom)
        
        if "{other_symptoms}" in template:
            other = [term for term in lay_terms if term != template.split()[0]]
            if other:
                template = template.replace("{other_symptoms}", random.choice(other))
        
        template = template.replace("{severity}", random.choice(severity_terms))
        template = template.replace("{time}", random.choice(time_terms))
        template = template.replace("{feeling}", random.choice(feeling_terms))
        
        print(f"Case {i+1}: {code} - {condition_data['name']}")
        print(f'  Patient: "{template}"')
        print()


def generate_synthetic_dataset(num_examples=20):
    """Generate a synthetic dataset for demonstration."""
    print(f"📊 Generating {num_examples} synthetic examples...")
    
    dataset = []
    
    for i in range(num_examples):
        code, condition_data = RespiratoryConditions.get_random_condition()
        
        # Generate symptoms
        primary_symptoms = condition_data["symptoms"]
        severity_symptoms = condition_data["severity_indicators"]
        lay_terms = condition_data["lay_terms"]
        
        num_symptoms = random.randint(3, 6)
        selected_symptoms = random.sample(primary_symptoms, min(3, len(primary_symptoms)))
        
        # Add some severity symptoms
        for symptom in severity_symptoms:
            if random.random() < 0.4 and len(selected_symptoms) < num_symptoms:
                selected_symptoms.append(symptom)
        
        # Generate mock complaint
        lay_sample = random.sample(lay_terms, min(2, len(lay_terms)))
        mock_complaint = f"I've been {lay_sample[0]} and {lay_sample[1] if len(lay_sample) > 1 else 'feeling awful'}."
        
        example = {
            "condition_code": code,
            "condition_name": condition_data["name"],
            "symptoms": selected_symptoms,
            "lay_terms_used": lay_sample,
            "patient_complaint": mock_complaint
        }
        
        dataset.append(example)
    
    # Save to file
    with open("synthetic_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Dataset saved to synthetic_dataset.json")
    
    # Show first few examples
    print("\nFirst 3 examples:")
    for i, example in enumerate(dataset[:3], 1):
        print(f"{i}. {example['condition_name']}")
        print(f"   Complaint: \"{example['patient_complaint']}\"")


def main():
    """Main demo function."""
    print("🏥 PHAITA - Pre-Hospital AI Triage Algorithm")
    print("🔬 Demonstration (No dependencies required)")
    print("=" * 60)
    print()
    
    demo_conditions()
    print("\n" + "="*60 + "\n")
    
    demo_symptom_generation()
    print("\n" + "="*60 + "\n")
    
    demo_complaint_simulation()
    print("\n" + "="*60 + "\n")
    
    generate_synthetic_dataset(10)
    
    print("\n" + "="*60)
    print("🎉 Demo completed!")
    print("\n📖 Next steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run full training: python cli.py train")
    print("  3. Test with models: python cli.py demo")


if __name__ == "__main__":
    main()