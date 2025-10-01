"""
Symptom and complaint generators using Bayesian networks and language models.
"""

import random
from typing import List, Optional, Dict
from .bayesian_network import BayesianSymptomNetwork
from ..data.icd_conditions import RespiratoryConditions


class SymptomGenerator:
    """
    Generates realistic symptom sets for medical conditions.
    """
    
    def __init__(self):
        """Initialize symptom generator with Bayesian network."""
        self.bayesian_network = BayesianSymptomNetwork()
    
    def generate_symptoms(
        self,
        condition_code: str,
        num_symptoms: Optional[int] = None
    ) -> List[str]:
        """
        Generate symptoms for a given condition.
        
        Args:
            condition_code: ICD-10 condition code
            num_symptoms: Number of symptoms to generate (None for random)
            
        Returns:
            List of symptoms
        """
        return self.bayesian_network.sample_symptoms(condition_code, num_symptoms)


class ComplaintGenerator:
    """
    Generates natural language patient complaints from symptoms.
    Uses template-based generation to simulate language model output.
    """
    
    def __init__(self):
        """Initialize complaint generator."""
        self.conditions = RespiratoryConditions.get_all_conditions()
        
        # Templates for generating complaints
        self.templates = [
            "I've been experiencing {symptoms} for the past {time}. It's really {severity}.",
            "Doctor, I have {symptoms} and I'm {feeling}. This started {time} ago.",
            "I'm having trouble with {symptoms}. It's been {severity} since {time}.",
            "Help, I can't stop {main_symptom}. I also have {other_symptoms}.",
            "I've been {main_symptom} and feeling {feeling}. It's been going on for {time}.",
            "My {symptoms} won't go away. Started {time} and getting {severity}.",
            "Really worried about {main_symptom}. Also experiencing {other_symptoms}.",
            "Can't seem to shake this {main_symptom}. {other_symptoms} too.",
        ]
        
        self.severity_terms = ["mild", "moderate", "severe", "terrible", "awful", "worse"]
        self.time_terms = ["a few hours", "yesterday", "two days", "this morning", "last night", "a week"]
        self.feeling_terms = ["worried", "scared", "exhausted", "panicked", "terrible", "awful"]
    
    def generate_complaint(
        self,
        symptoms: List[str],
        condition_code: str,
        use_lay_terms: bool = True
    ) -> str:
        """
        Generate a natural language patient complaint.
        
        Args:
            symptoms: List of medical symptoms
            condition_code: ICD-10 condition code
            use_lay_terms: Whether to use lay language
            
        Returns:
            Natural language complaint string
        """
        if not symptoms:
            return "I'm not feeling well."
        
        # Get lay terms if available
        if use_lay_terms and condition_code in self.conditions:
            lay_terms = self.conditions[condition_code]["lay_terms"]
            if lay_terms:
                # Use lay terms for some symptoms
                display_symptoms = []
                for symptom in symptoms[:3]:  # Use first 3 symptoms
                    if random.random() < 0.7 and lay_terms:
                        display_symptoms.append(random.choice(lay_terms))
                    else:
                        display_symptoms.append(symptom.replace('_', ' '))
            else:
                display_symptoms = [s.replace('_', ' ') for s in symptoms[:3]]
        else:
            display_symptoms = [s.replace('_', ' ') for s in symptoms[:3]]
        
        # Select template
        template = random.choice(self.templates)
        
        # Fill in template
        complaint = template
        
        if "{symptoms}" in complaint:
            symptoms_text = " and ".join(display_symptoms[:2])
            complaint = complaint.replace("{symptoms}", symptoms_text)
        
        if "{main_symptom}" in complaint:
            main = display_symptoms[0] if display_symptoms else "not feeling well"
            complaint = complaint.replace("{main_symptom}", main)
        
        if "{other_symptoms}" in complaint:
            other = display_symptoms[1] if len(display_symptoms) > 1 else "feeling unwell"
            complaint = complaint.replace("{other_symptoms}", other)
        
        complaint = complaint.replace("{severity}", random.choice(self.severity_terms))
        complaint = complaint.replace("{time}", random.choice(self.time_terms))
        complaint = complaint.replace("{feeling}", random.choice(self.feeling_terms))
        
        return complaint
    
    def generate_multiple_complaints(
        self,
        condition_code: str,
        num_complaints: int = 5
    ) -> List[str]:
        """
        Generate multiple complaints for a condition.
        
        Args:
            condition_code: ICD-10 condition code
            num_complaints: Number of complaints to generate
            
        Returns:
            List of complaint strings
        """
        complaints = []
        symptom_gen = SymptomGenerator()
        
        for _ in range(num_complaints):
            symptoms = symptom_gen.generate_symptoms(condition_code)
            complaint = self.generate_complaint(symptoms, condition_code)
            complaints.append(complaint)
        
        return complaints
