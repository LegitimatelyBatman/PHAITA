"""
Enhanced Bayesian network with age, severity, and rare presentation modeling.
"""

import random
from typing import List, Tuple, Dict, Optional
from .bayesian_network import BayesianSymptomNetwork
from ..data.icd_conditions import RespiratoryConditions


class EnhancedBayesianNetwork:
    """
    Enhanced Bayesian network with age-specific, severity-specific, and rare presentation modeling.
    """
    
    def __init__(self):
        """Initialize enhanced Bayesian network."""
        self.base_network = BayesianSymptomNetwork()
        self.conditions = RespiratoryConditions.get_all_conditions()
        
        # Age group modifiers
        self.age_modifiers = {
            "child": {
                "symptom_prob_multiplier": 1.2,
                "severity_prob_multiplier": 0.8,
                "specific_symptoms": ["irritability", "poor_feeding", "nasal_flaring"]
            },
            "adult": {
                "symptom_prob_multiplier": 1.0,
                "severity_prob_multiplier": 1.0,
                "specific_symptoms": []
            },
            "elderly": {
                "symptom_prob_multiplier": 0.9,
                "severity_prob_multiplier": 1.3,
                "specific_symptoms": ["confusion", "weakness", "fatigue"]
            }
        }
        
        # Severity modifiers
        self.severity_modifiers = {
            "mild": {
                "symptom_prob_multiplier": 0.6,
                "severity_prob_multiplier": 0.2,
                "num_symptoms": (2, 4)
            },
            "moderate": {
                "symptom_prob_multiplier": 1.0,
                "severity_prob_multiplier": 0.5,
                "num_symptoms": (3, 5)
            },
            "severe": {
                "symptom_prob_multiplier": 1.2,
                "severity_prob_multiplier": 1.5,
                "num_symptoms": (4, 7)
            }
        }
        
        # Rare presentation cases
        self.rare_presentations = {
            "J45.9": [  # Asthma
                {
                    "name": "Silent chest asthma",
                    "symptoms": ["minimal_wheezing", "severe_dyspnea", "anxiety", "tachycardia"],
                    "probability": 0.02
                },
                {
                    "name": "Cough-variant asthma",
                    "symptoms": ["chronic_cough", "no_wheezing", "chest_tightness"],
                    "probability": 0.05
                }
            ],
            "J18.9": [  # Pneumonia
                {
                    "name": "Walking pneumonia",
                    "symptoms": ["mild_cough", "fatigue", "low_grade_fever"],
                    "probability": 0.1
                }
            ],
            "J81.0": [  # Pulmonary edema
                {
                    "name": "Flash pulmonary edema",
                    "symptoms": ["sudden_severe_dyspnea", "anxiety", "diaphoresis"],
                    "probability": 0.03
                }
            ]
        }
        
        # Evidence sources (mock medical literature references)
        self.evidence_sources = {
            "J45.9": {
                "wheezing": "GINA Guidelines 2023",
                "dyspnea": "GINA Guidelines 2023",
                "chest_tightness": "GINA Guidelines 2023",
                "cough": "GINA Guidelines 2023"
            },
            "J18.9": {
                "cough": "CAP Guidelines 2019",
                "fever": "CAP Guidelines 2019",
                "chest_pain": "CAP Guidelines 2019",
                "dyspnea": "CAP Guidelines 2019"
            },
            "J44.9": {
                "chronic_cough": "GOLD 2023",
                "dyspnea": "GOLD 2023",
                "sputum_production": "GOLD 2023"
            }
        }
    
    def sample_symptoms(
        self,
        condition_code: str,
        age_group: str = "adult",
        severity: str = "moderate",
        include_rare: bool = False
    ) -> Tuple[List[str], Dict]:
        """
        Sample symptoms with age, severity, and rare presentation considerations.
        
        Args:
            condition_code: ICD-10 condition code
            age_group: Age group (child, adult, elderly)
            severity: Severity level (mild, moderate, severe)
            include_rare: Whether to include rare presentations
            
        Returns:
            Tuple of (symptoms, metadata)
        """
        metadata = {
            "age_group": age_group,
            "severity": severity,
            "presentation_type": "standard"
        }
        
        # Check for rare presentation
        if include_rare and condition_code in self.rare_presentations:
            rare_cases = self.rare_presentations[condition_code]
            for case in rare_cases:
                if random.random() < case["probability"]:
                    metadata["presentation_type"] = "rare"
                    metadata["case_name"] = case["name"]
                    return case["symptoms"].copy(), metadata
        
        # Get base condition data
        if condition_code not in self.conditions:
            raise ValueError(f"Unknown condition code: {condition_code}")
        
        condition_data = self.conditions[condition_code]
        primary_symptoms = condition_data["symptoms"]
        severity_symptoms = condition_data["severity_indicators"]
        
        # Apply severity modifier
        severity_mod = self.severity_modifiers.get(severity, self.severity_modifiers["moderate"])
        num_symptoms_range = severity_mod["num_symptoms"]
        num_symptoms = random.randint(*num_symptoms_range)
        
        # Apply age modifier
        age_mod = self.age_modifiers.get(age_group, self.age_modifiers["adult"])
        
        # Calculate adjusted probabilities
        primary_prob = self.base_network.primary_symptom_prob * severity_mod["symptom_prob_multiplier"] * age_mod["symptom_prob_multiplier"]
        severity_prob = self.base_network.severity_symptom_prob * severity_mod["severity_prob_multiplier"] * age_mod["severity_prob_multiplier"]
        
        sampled = []
        
        # Sample primary symptoms
        for symptom in primary_symptoms:
            if random.random() < primary_prob:
                sampled.append(symptom)
        
        # Sample severity symptoms
        for symptom in severity_symptoms:
            if random.random() < severity_prob and len(sampled) < num_symptoms:
                sampled.append(symptom)
        
        # Add age-specific symptoms
        if age_mod["specific_symptoms"] and random.random() < 0.3:
            age_symptom = random.choice(age_mod["specific_symptoms"])
            if age_symptom not in sampled:
                sampled.append(age_symptom)
        
        # Ensure minimum symptoms
        if not sampled:
            sampled.append(random.choice(primary_symptoms))
        
        # Trim to desired number
        if len(sampled) > num_symptoms:
            sampled = random.sample(sampled, num_symptoms)
        
        return sampled, metadata
    
    def get_evidence_sources(self, condition_code: str) -> Dict[str, str]:
        """
        Get evidence sources for condition symptoms.
        
        Args:
            condition_code: ICD-10 condition code
            
        Returns:
            Dictionary mapping symptoms to evidence sources
        """
        return self.evidence_sources.get(condition_code, {})
    
    def get_rare_presentations(self, condition_code: str) -> List[Dict]:
        """
        Get rare presentations for a condition.
        
        Args:
            condition_code: ICD-10 condition code
            
        Returns:
            List of rare presentation dictionaries
        """
        return self.rare_presentations.get(condition_code, [])


def create_enhanced_bayesian_network() -> EnhancedBayesianNetwork:
    """Create an EnhancedBayesianNetwork instance."""
    return EnhancedBayesianNetwork()
