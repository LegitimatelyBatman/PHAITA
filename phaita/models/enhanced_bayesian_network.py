"""
Enhanced Bayesian network with age, severity, rare presentation, and comorbidity modeling.
"""

import random
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from .bayesian_network import BayesianSymptomNetwork
from ..data.icd_conditions import RespiratoryConditions


class EnhancedBayesianNetwork:
    """
    Enhanced Bayesian network with age-specific, severity-specific, rare presentation, and comorbidity modeling.
    
    Supports modeling of:
    - Age-specific symptom patterns (child, adult, elderly)
    - Severity-specific presentations (mild, moderate, severe)
    - Rare presentations for conditions
    - Comorbidity effects on symptom probabilities
    - Cross-condition interactions (e.g., asthma + COPD)
    """
    
    def __init__(self):
        """Initialize enhanced Bayesian network."""
        self.base_network = BayesianSymptomNetwork()
        self.conditions = RespiratoryConditions.get_all_conditions()
        
        # Load comorbidity effects from config file
        self._load_comorbidity_effects()
        
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
    
    def _load_comorbidity_effects(self):
        """Load comorbidity effects from YAML configuration file."""
        config_path = Path(__file__).parent.parent.parent / "config" / "comorbidity_effects.yaml"
        
        # Default fallback modifiers if config file is not found
        default_modifiers = {
            "diabetes": {"fatigue": 1.3, "infection_risk": 1.5},
            "hypertension": {"dyspnea": 1.2, "chest_pain": 1.4},
            "obesity": {"shortness_of_breath": 1.5, "exercise_intolerance": 1.3}
        }
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                # Extract comorbidity modifiers
                self.comorbidity_modifiers = {}
                for comorbidity, data in config_data.items():
                    if isinstance(data, dict) and 'symptom_modifiers' in data:
                        self.comorbidity_modifiers[comorbidity] = {
                            'symptom_modifiers': data['symptom_modifiers'],
                            'specific_symptoms': data.get('specific_symptoms', []),
                            'probability': data.get('probability', 0.3)
                        }
                
                # Extract interaction effects
                self.interaction_effects = config_data.get('interactions', {})
                self.max_symptom_probability = config_data.get('max_probability', 0.95)
            else:
                # Use defaults if config not found
                self.comorbidity_modifiers = {k: {'symptom_modifiers': v, 'specific_symptoms': [], 'probability': 0.3} 
                                             for k, v in default_modifiers.items()}
                self.interaction_effects = {}
                self.max_symptom_probability = 0.95
        except Exception as e:
            # Fallback to defaults on any error
            self.comorbidity_modifiers = {k: {'symptom_modifiers': v, 'specific_symptoms': [], 'probability': 0.3} 
                                         for k, v in default_modifiers.items()}
            self.interaction_effects = {}
            self.max_symptom_probability = 0.95
    
    def sample_symptoms(
        self,
        condition_code: str,
        comorbidities: Optional[List[str]] = None,
        age_group: str = "adult",
        severity: str = "moderate",
        include_rare: bool = False
    ) -> Tuple[List[str], Dict]:
        """
        Sample symptoms with age, severity, comorbidity, and rare presentation considerations.
        
        Comorbidities modify symptom probabilities based on clinical evidence. For example:
        - Diabetes increases fatigue (1.3x) and infection risk (1.5x)
        - Hypertension increases dyspnea (1.2x) and chest pain (1.4x)
        - Obesity increases shortness of breath (1.5x) and exercise intolerance (1.3x)
        
        Cross-condition interactions are also modeled:
        - Asthma + COPD: chronic cough probability increases to 0.9 (ACOS)
        - Multiple comorbidities compound their effects multiplicatively
        
        All symptom probabilities are capped at 0.95 to maintain realistic distributions.
        
        Args:
            condition_code: ICD-10 condition code
            comorbidities: List of comorbidity names (e.g., ["diabetes", "hypertension"])
            age_group: Age group (child, adult, elderly)
            severity: Severity level (mild, moderate, severe)
            include_rare: Whether to include rare presentations
            
        Returns:
            Tuple of (symptoms, metadata)
        """
        comorbidities = comorbidities or []
        metadata = {
            "age_group": age_group,
            "severity": severity,
            "presentation_type": "standard"
        }
        
        if comorbidities:
            metadata["comorbidities"] = comorbidities
        
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
        
        # Build symptom probability map with comorbidity adjustments
        symptom_probs = {}
        
        # Calculate base probabilities for primary symptoms
        for symptom in primary_symptoms:
            symptom_probs[symptom] = primary_prob
        
        # Add severity symptoms with their probabilities
        for symptom in severity_symptoms:
            symptom_probs[symptom] = severity_prob
        
        # Apply comorbidity modifiers to symptom probabilities
        comorbidity_specific_symptoms = []
        if comorbidities:
            for comorbidity in comorbidities:
                if comorbidity in self.comorbidity_modifiers:
                    comorbidity_data = self.comorbidity_modifiers[comorbidity]
                    modifiers = comorbidity_data['symptom_modifiers']
                    
                    # Apply modifiers to existing symptoms
                    for symptom, multiplier in modifiers.items():
                        # Normalize symptom name (replace _ with space and vice versa for matching)
                        symptom_normalized = symptom.replace('_', ' ')
                        
                        # Check if this symptom exists in our symptom list
                        for existing_symptom in list(symptom_probs.keys()):
                            existing_normalized = existing_symptom.replace('_', ' ')
                            if symptom_normalized == existing_normalized or symptom == existing_symptom:
                                # Apply multiplier and cap at max probability
                                symptom_probs[existing_symptom] = min(
                                    symptom_probs[existing_symptom] * multiplier,
                                    self.max_symptom_probability
                                )
                    
                    # Add comorbidity-specific symptoms
                    if comorbidity_data['specific_symptoms'] and random.random() < comorbidity_data['probability']:
                        specific = random.choice(comorbidity_data['specific_symptoms'])
                        if specific not in comorbidity_specific_symptoms:
                            comorbidity_specific_symptoms.append(specific)
        
        # Check for cross-condition interactions
        if comorbidities:
            self._apply_interaction_effects(condition_code, comorbidities, symptom_probs)
        
        sampled = []
        
        # Sample symptoms based on adjusted probabilities
        for symptom, prob in symptom_probs.items():
            if random.random() < prob:
                sampled.append(symptom)
        
        # Add comorbidity-specific symptoms
        for specific_symptom in comorbidity_specific_symptoms:
            if specific_symptom not in sampled:
                sampled.append(specific_symptom)
        
        # Add age-specific symptoms
        if age_mod["specific_symptoms"] and random.random() < 0.3:
            age_symptom = random.choice(age_mod["specific_symptoms"])
            if age_symptom not in sampled:
                sampled.append(age_symptom)
        
        # Ensure minimum symptoms
        if not sampled:
            sampled.append(random.choice(primary_symptoms))
        
        # Trim to desired number, but preserve high-probability symptoms from interactions
        # (e.g., chronic_cough in ACOS should not be randomly dropped)
        if len(sampled) > num_symptoms:
            # Identify high-priority symptoms (prob >= 0.85 from interactions)
            high_priority = [s for s in sampled if symptom_probs.get(s, 0) >= 0.85]
            other_symptoms = [s for s in sampled if symptom_probs.get(s, 0) < 0.85]
            
            # Keep all high-priority symptoms and sample from others to fill remaining slots
            remaining_slots = num_symptoms - len(high_priority)
            if remaining_slots > 0 and other_symptoms:
                sampled = high_priority + random.sample(other_symptoms, min(remaining_slots, len(other_symptoms)))
            else:
                sampled = high_priority[:num_symptoms]
        
        return sampled, metadata
    
    def _apply_interaction_effects(self, condition_code: str, comorbidities: List[str], symptom_probs: Dict[str, float]):
        """
        Apply cross-condition interaction effects on symptom probabilities.
        
        Examples:
        - Asthma + COPD: chronic cough probability → 0.9 (ACOS syndrome)
        - Heart failure + COPD: additive dyspnea effects
        
        Args:
            condition_code: ICD-10 condition code
            comorbidities: List of comorbidities
            symptom_probs: Dictionary of symptom probabilities to modify in-place
        """
        # Check for Asthma-COPD Overlap Syndrome (ACOS)
        if condition_code == "J45.9" and "copd" in [c.lower() for c in comorbidities]:
            # ACOS - documented interaction from GINA/GOLD guidelines
            # Add chronic_cough which may not be in base symptoms
            symptom_probs["chronic_cough"] = 0.9
            symptom_probs["wheezing"] = max(symptom_probs.get("wheezing", 0.5), 0.88)
            if "dyspnea" in symptom_probs:
                symptom_probs["dyspnea"] = min(symptom_probs["dyspnea"] * 1.2, 0.92)
            if "shortness_of_breath" in symptom_probs:
                symptom_probs["shortness_of_breath"] = min(symptom_probs["shortness_of_breath"] * 1.2, 0.92)
            # Also boost sputum_production if it exists
            symptom_probs["sputum_production"] = max(symptom_probs.get("sputum_production", 0.5), 0.85)
        
        elif condition_code == "J44.9" and "copd" not in [c.lower() for c in comorbidities]:
            # Check if COPD condition and asthma comorbidity
            for comorbidity in comorbidities:
                if "asthma" in comorbidity.lower():
                    symptom_probs["chronic_cough"] = 0.9
                    symptom_probs["wheezing"] = max(symptom_probs.get("wheezing", 0.5), 0.88)
                    if "dyspnea" in symptom_probs:
                        symptom_probs["dyspnea"] = min(symptom_probs["dyspnea"] * 1.2, 0.92)
                    break
        
        # Apply other interaction effects from config
        for interaction_name, interaction_data in self.interaction_effects.items():
            if not isinstance(interaction_data, dict):
                continue
                
            # Check if conditions match
            if 'conditions' in interaction_data and condition_code in interaction_data['conditions']:
                # Check if comorbidity matches
                if 'comorbidity' in interaction_data:
                    if interaction_data['comorbidity'] in comorbidities:
                        # Apply interaction modifiers
                        for symptom, modifier in interaction_data.get('symptom_modifiers', {}).items():
                            # If modifier is > 1, treat as multiplier; if < 1, treat as absolute probability
                            if modifier > 1.0:
                                symptom_probs[symptom] = min(
                                    symptom_probs.get(symptom, 0.5) * modifier,
                                    self.max_symptom_probability
                                )
                            else:
                                # Absolute probability for cross-condition symptoms
                                symptom_probs[symptom] = modifier
    
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
