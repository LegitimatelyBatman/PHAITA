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
        
        # Templates for generating complaints with proper grammar
        # Use {symptoms_experiencing} for "experiencing X" form
        # Use {main_symptom_gerund} for gerund form (e.g., "wheezing")
        # Use {main_symptom_phrase} for phrase form (e.g., "trouble breathing")
        self.templates = [
            "I've been experiencing {symptoms} for the past {time}. It's really {severity}.",
            "Doctor, I have {symptoms} and I'm {feeling}. This started {time} ago.",
            "I'm having {main_symptom_phrase}. It's been {severity} since {time}.",
            "Help, I {main_symptom_action}. I also have {other_symptom_phrase}.",
            "I've been {main_symptom_gerund} and feeling {feeling}. It's been going on for {time}.",
            "My {symptoms_phrase_form} won't go away. Started {time} and getting {severity}.",
            "Really worried about {main_symptom_phrase}. Also experiencing {other_symptom_phrase}.",
            "Can't seem to shake this {main_symptom_noun}. {other_symptom_phrase} too.",
        ]
        
        self.severity_terms = ["mild", "moderate", "severe", "terrible", "awful", "worse"]
        self.time_terms = ["a few hours", "yesterday", "two days", "this morning", "last night", "a week"]
        self.feeling_terms = ["worried", "scared", "exhausted", "panicked", "terrible", "awful"]
        
        # Grammar rules for symptom transformation
        self.symptom_grammar_rules = {
            # Map symptoms to their proper grammatical forms
            "wheezing": {
                "gerund": "wheezing",
                "noun": "wheezing",
                "phrase": "my wheezing",
                "action": "can't stop wheezing"
            },
            "shortness_of_breath": {
                "gerund": "having shortness of breath",
                "noun": "breathlessness", 
                "phrase": "shortness of breath",
                "action": "can't catch my breath"
            },
            "difficulty_breathing": {
                "gerund": "having difficulty breathing",
                "noun": "breathing difficulty",
                "phrase": "difficulty breathing",
                "action": "can't breathe properly"
            },
            "coughing": {
                "gerund": "coughing",
                "noun": "cough",
                "phrase": "my cough",
                "action": "can't stop coughing"
            },
            "cough": {
                "gerund": "coughing",
                "noun": "cough",
                "phrase": "my cough",
                "action": "can't stop coughing"
            },
            "chest_pain": {
                "gerund": "experiencing chest pain",
                "noun": "chest pain",
                "phrase": "chest pain",
                "action": "have sharp chest pain"
            },
            "chest_tightness": {
                "gerund": "experiencing chest tightness",
                "noun": "chest tightness",
                "phrase": "tight chest",
                "action": "feel chest tightness"
            },
            "tight_chest": {
                "gerund": "experiencing chest tightness",
                "noun": "chest tightness",
                "phrase": "tight chest",
                "action": "feel tightness in my chest"
            },
            "fever": {
                "gerund": "running a fever",
                "noun": "fever",
                "phrase": "my fever",
                "action": "have a fever"
            },
            "breathless": {
                "gerund": "feeling breathless",
                "noun": "breathlessness",
                "phrase": "breathlessness",
                "action": "feel breathless"
            },
            "breathlessness": {
                "gerund": "feeling breathless",
                "noun": "breathlessness",
                "phrase": "breathlessness",
                "action": "feel breathless"
            },
            "gasping_for_air": {
                "gerund": "gasping for air",
                "noun": "breathlessness",
                "phrase": "gasping for air",
                "action": "can't stop gasping for air"
            },
            "gasping for air": {
                "gerund": "gasping for air",
                "noun": "breathlessness",
                "phrase": "gasping for air",
                "action": "can't stop gasping for air"
            },
            "can't breathe": {
                "gerund": "having trouble breathing",
                "noun": "breathing difficulty",
                "phrase": "trouble breathing",
                "action": "can't breathe"
            },
            "can't catch my breath": {
                "gerund": "having trouble catching my breath",
                "noun": "breathlessness",
                "phrase": "trouble catching my breath",
                "action": "can't catch my breath"
            },
            "wheezy": {
                "gerund": "feeling wheezy",
                "noun": "wheezing",
                "phrase": "wheezing",
                "action": "feel wheezy"
            }
        }
    
    def _get_symptom_form(self, symptom: str, form: str) -> str:
        """
        Get the grammatically correct form of a symptom.
        
        Args:
            symptom: Raw symptom string (e.g., "wheezing", "shortness_of_breath")
            form: Desired grammatical form ("gerund", "noun", "phrase", "action")
            
        Returns:
            Grammatically correct symptom form
        """
        # Normalize symptom
        symptom_key = symptom.lower().replace(' ', '_')
        
        # Also try with spaces for lay terms
        symptom_key_spaced = symptom.lower()
        
        # Check if we have grammar rules for this symptom
        if symptom_key in self.symptom_grammar_rules:
            return self.symptom_grammar_rules[symptom_key].get(form, symptom.replace('_', ' '))
        if symptom_key_spaced in self.symptom_grammar_rules:
            return self.symptom_grammar_rules[symptom_key_spaced].get(form, symptom)
        
        # Fallback: apply default grammar rules
        symptom_clean = symptom.replace('_', ' ')
        
        if form == "gerund":
            # Check if it's already a gerund (ends in -ing)
            if symptom_clean.endswith('ing'):
                return symptom_clean
            # Check if it starts with "can't" - convert to proper form
            if symptom_clean.startswith("can't"):
                base = symptom_clean.replace("can't ", "")
                return f"having trouble {base}"
            # For phrases with "of", prepend "having"
            if ' of ' in symptom_clean or symptom_clean.startswith('difficulty'):
                return f"having {symptom_clean}"
            # For most symptoms, add "experiencing"
            return f"experiencing {symptom_clean}"
        elif form == "noun":
            # Handle "can't X" phrases
            if symptom_clean.startswith("can't"):
                return "breathing difficulty"
            # Remove trailing -ing if present and convert to noun form
            if symptom_clean.endswith('ing'):
                return symptom_clean  # Keep as-is (e.g., "wheezing" is both verb and noun)
            # Handle adjectives
            if symptom_clean in ['breathless', 'wheezy', 'dizzy']:
                return symptom_clean + 'ness'
            return symptom_clean
        elif form == "phrase":
            # Handle "can't X" phrases
            if symptom_clean.startswith("can't"):
                base = symptom_clean.replace("can't ", "")
                return f"trouble {base if base else 'breathing'}"
            return symptom_clean
        elif form == "action":
            # Create action phrase (e.g., "can't stop X" or "have X")
            if symptom_clean.startswith("can't"):
                return symptom_clean  # Already an action
            if symptom_clean.endswith('ing'):
                return f"can't stop {symptom_clean}"
            return f"have {symptom_clean}"
        
        return symptom_clean
    
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
                    if random.random() < 0.5 and lay_terms:
                        display_symptoms.append(random.choice(lay_terms))
                    else:
                        display_symptoms.append(symptom.replace('_', ' '))
            else:
                display_symptoms = [s.replace('_', ' ') for s in symptoms[:3]]
        else:
            display_symptoms = [s.replace('_', ' ') for s in symptoms[:3]]
        
        # Select template
        template = random.choice(self.templates)
        
        # Fill in template with grammatically correct forms
        complaint = template
        
        # Handle {symptoms} - basic list
        if "{symptoms}" in complaint:
            symptoms_text = " and ".join(display_symptoms[:2])
            complaint = complaint.replace("{symptoms}", symptoms_text)
        
        # Handle {symptoms_phrase_form} - symptoms in phrase form (for "My X" constructions)
        if "{symptoms_phrase_form}" in complaint:
            symptom_phrases = [self._get_symptom_form(s, "phrase") for s in display_symptoms[:2]]
            symptoms_phrase = " and ".join(symptom_phrases)
            complaint = complaint.replace("{symptoms_phrase_form}", symptoms_phrase)
        
        # Handle {main_symptom_gerund} - gerund form
        if "{main_symptom_gerund}" in complaint:
            main = display_symptoms[0] if display_symptoms else "not feeling well"
            main_gerund = self._get_symptom_form(main, "gerund")
            complaint = complaint.replace("{main_symptom_gerund}", main_gerund)
        
        # Handle {main_symptom_noun} - noun form
        if "{main_symptom_noun}" in complaint:
            main = display_symptoms[0] if display_symptoms else "illness"
            main_noun = self._get_symptom_form(main, "noun")
            complaint = complaint.replace("{main_symptom_noun}", main_noun)
        
        # Handle {main_symptom_phrase} - phrase form
        if "{main_symptom_phrase}" in complaint:
            main = display_symptoms[0] if display_symptoms else "not feeling well"
            main_phrase = self._get_symptom_form(main, "phrase")
            complaint = complaint.replace("{main_symptom_phrase}", main_phrase)
        
        # Handle {main_symptom_action} - action form
        if "{main_symptom_action}" in complaint:
            main = display_symptoms[0] if display_symptoms else "feel unwell"
            main_action = self._get_symptom_form(main, "action")
            complaint = complaint.replace("{main_symptom_action}", main_action)
        
        # Handle {other_symptom_phrase} - other symptoms in phrase form
        if "{other_symptom_phrase}" in complaint:
            other = display_symptoms[1] if len(display_symptoms) > 1 else "feeling unwell"
            other_phrase = self._get_symptom_form(other, "phrase")
            complaint = complaint.replace("{other_symptom_phrase}", other_phrase)
        
        # Replace standard placeholders
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
