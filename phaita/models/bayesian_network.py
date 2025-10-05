"""Bayesian symptom network for probabilistic symptom generation."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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


if TORCH_AVAILABLE:
    class LearnableBayesianSymptomNetwork(nn.Module):
        """
        Learnable Bayesian network with neural network weights for symptom sampling.
        Uses PyTorch parameters instead of hardcoded probabilities.
        """
        
        def __init__(self, conditions: Optional[Dict[str, Dict]] = None, device: Optional[str] = None):
            """
            Initialize learnable Bayesian symptom network.
            
            Args:
                conditions: Dictionary of medical conditions
                device: Device for PyTorch tensors ('cpu' or 'cuda')
            """
            super().__init__()
            
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.conditions: Dict[str, Dict] = {}
            
            # Initialize learnable weights
            # Primary symptom probability (initialized to 0.8)
            self.primary_symptom_logit = nn.Parameter(
                torch.tensor(1.386, device=self.device)  # logit(0.8) ≈ 1.386
            )
            
            # Severity symptom probability (initialized to 0.4)
            self.severity_symptom_logit = nn.Parameter(
                torch.tensor(-0.405, device=self.device)  # logit(0.4) ≈ -0.405
            )
            
            # Per-condition weight adjustments (initialized to 0, meaning no adjustment)
            self.condition_weights: Optional[nn.Parameter] = None
            
            self.reload(conditions=conditions)
        
        def reload(self, conditions: Optional[Dict[str, Dict]] = None) -> None:
            """Refresh the network with the latest condition catalogue."""
            if conditions is None:
                conditions = RespiratoryConditions.get_all_conditions()
            self.conditions = conditions
            
            # Initialize per-condition weights if not already done
            num_conditions = len(self.conditions)
            if self.condition_weights is None or self.condition_weights.shape[0] != num_conditions:
                # Shape: [num_conditions, 2] for primary and severity adjustments
                self.condition_weights = nn.Parameter(
                    torch.zeros(num_conditions, 2, device=self.device)
                )
        
        def get_probabilities(self, condition_code: str) -> tuple[float, float]:
            """
            Get symptom probabilities for a condition with learnable adjustments.
            
            Args:
                condition_code: ICD-10 condition code
                
            Returns:
                Tuple of (primary_prob, severity_prob)
            """
            # Base probabilities from learnable parameters
            primary_prob = torch.sigmoid(self.primary_symptom_logit)
            severity_prob = torch.sigmoid(self.severity_symptom_logit)
            
            # Apply per-condition adjustments
            if condition_code in self.conditions and self.condition_weights is not None:
                condition_idx = list(self.conditions.keys()).index(condition_code)
                adjustments = torch.sigmoid(self.condition_weights[condition_idx])
                
                # Adjustments are multiplicative modifiers
                primary_prob = torch.clamp(primary_prob * (0.5 + adjustments[0]), 0.1, 0.99)
                severity_prob = torch.clamp(severity_prob * (0.5 + adjustments[1]), 0.05, 0.9)
            
            return primary_prob.item(), severity_prob.item()
        
        def sample_symptoms(
            self,
            condition_code: str,
            num_symptoms: Optional[int] = None
        ) -> List[str]:
            """
            Sample symptoms for a given condition using learnable probabilities.
            
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
            
            # Get learnable probabilities
            primary_prob, severity_prob = self.get_probabilities(condition_code)
            
            # Determine number of symptoms to sample
            if num_symptoms is None:
                num_symptoms = random.randint(3, 6)
            
            sampled = []
            
            # Sample primary symptoms with learned probability
            for symptom in primary_symptoms:
                if random.random() < primary_prob:
                    sampled.append(symptom)
            
            # Sample severity symptoms with learned probability
            for symptom in severity_symptoms:
                if random.random() < severity_prob and len(sampled) < num_symptoms:
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
            
            primary_prob, severity_prob = self.get_probabilities(condition_code)
            
            if symptom in normalized_primary:
                return primary_prob
            elif symptom in normalized_severity:
                return severity_prob
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
            primary_prob, severity_prob = self.get_probabilities(condition_code)
            
            probs = {}
            
            for symptom in condition_data["symptoms"]:
                probs[symptom] = primary_prob
            
            for symptom in condition_data["severity_indicators"]:
                probs[symptom] = severity_prob
            
            return probs
