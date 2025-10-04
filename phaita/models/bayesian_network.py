"""Bayesian symptom network for probabilistic symptom generation."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..data.icd_conditions import RespiratoryConditions
from ..utils.text import normalize_symptom


class BayesianSymptomNetwork:
    """
    Bayesian network for modeling symptom relationships and generating realistic symptom sets.
    """

    def __init__(self, conditions: Optional[Dict[str, Dict]] = None):
        """Initialize the Bayesian symptom network."""
        self.conditions: Dict[str, Dict] = {}
        self.reload(conditions=conditions)

    def reload(self, conditions: Optional[Dict[str, Dict]] = None) -> None:
        """Refresh the network with the latest condition catalogue."""

        if conditions is None:
            conditions = RespiratoryConditions.get_all_conditions()
        self.conditions = conditions

        # Probability parameters
        self.primary_symptom_prob = 0.8  # High probability for primary symptoms
        self.severity_symptom_prob = 0.4  # Lower for severity indicators
        
    def sample_symptoms(
        self,
        condition_code: str,
        num_symptoms: Optional[int] = None
    ) -> List[str]:
        """
        Sample symptoms for a given condition using probabilistic sampling.
        
        Args:
            condition_code: ICD-10 condition code
            num_symptoms: Number of symptoms to sample (None for random 3-6)
            
        Returns:
            List of sampled symptoms
        """
        if condition_code not in self.conditions:
            raise ValueError(f"Unknown condition code: {condition_code}")
        
        condition_data = self.conditions[condition_code]
        primary_symptoms = condition_data["symptoms"]
        severity_symptoms = condition_data["severity_indicators"]
        
        # Determine number of symptoms to sample
        if num_symptoms is None:
            num_symptoms = random.randint(3, 6)
        
        sampled = []
        
        # Sample primary symptoms with higher probability
        for symptom in primary_symptoms:
            if random.random() < self.primary_symptom_prob:
                sampled.append(symptom)
        
        # Sample severity symptoms with lower probability
        for symptom in severity_symptoms:
            if random.random() < self.severity_symptom_prob and len(sampled) < num_symptoms:
                sampled.append(symptom)
        
        # Ensure we have at least one symptom
        if not sampled:
            sampled.append(random.choice(primary_symptoms))
        
        # If we have too many, randomly select subset
        if len(sampled) > num_symptoms:
            sampled = random.sample(sampled, num_symptoms)
        
        # If we have too few, add more from primary symptoms
        while len(sampled) < min(num_symptoms, len(primary_symptoms) + len(severity_symptoms)):
            remaining = [s for s in primary_symptoms + severity_symptoms if s not in sampled]
            if remaining:
                sampled.append(random.choice(remaining))
            else:
                break
        
        return sampled
    
    def get_symptom_probability(self, condition_code: str, symptom: str) -> float:
        """
        Get the probability of a symptom given a condition.
        
        Args:
            condition_code: ICD-10 condition code
            symptom: Symptom name
            
        Returns:
            Probability (0.0 to 1.0)
        """
        # Normalize symptom for consistent matching
        symptom = normalize_symptom(symptom)
        
        if condition_code not in self.conditions:
            return 0.0
        
        condition_data = self.conditions[condition_code]
        
        # Normalize condition symptoms for comparison
        normalized_primary = [normalize_symptom(s) for s in condition_data["symptoms"]]
        normalized_severity = [normalize_symptom(s) for s in condition_data["severity_indicators"]]
        
        if symptom in normalized_primary:
            return self.primary_symptom_prob
        elif symptom in normalized_severity:
            return self.severity_symptom_prob
        else:
            return 0.0
    
    def get_conditional_probabilities(self, condition_code: str) -> Dict[str, float]:
        """
        Get all symptom probabilities for a condition.
        
        Args:
            condition_code: ICD-10 condition code
            
        Returns:
            Dictionary mapping symptoms to probabilities
        """
        if condition_code not in self.conditions:
            return {}
        
        condition_data = self.conditions[condition_code]
        probs = {}
        
        for symptom in condition_data["symptoms"]:
            probs[symptom] = self.primary_symptom_prob
        
        for symptom in condition_data["severity_indicators"]:
            probs[symptom] = self.severity_symptom_prob
        
        return probs
