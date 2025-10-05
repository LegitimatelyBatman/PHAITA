"""
Medical accuracy loss functions for training learnable Bayesian networks.
Ensures symptom-condition alignment with medical knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import logging

from ..data.icd_conditions import RespiratoryConditions


class MedicalAccuracyLoss(nn.Module):
    """
    Loss function for measuring medical accuracy of symptom sampling.
    Guides learnable Bayesian network to maintain medically plausible symptom distributions.
    
    The loss consists of:
    1. Symptom-condition alignment: Ensures generated symptoms match expected patterns
    2. Probability constraint: Keeps probabilities in reasonable ranges
    3. Diversity penalty: Prevents all conditions from having identical distributions
    """
    
    def __init__(
        self,
        alignment_weight: float = 1.0,
        constraint_weight: float = 0.5,
        diversity_weight: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize medical accuracy loss.
        
        Args:
            alignment_weight: Weight for symptom-condition alignment loss
            constraint_weight: Weight for probability constraint loss
            diversity_weight: Weight for diversity penalty
            device: Device for computations
        """
        super().__init__()
        self.alignment_weight = alignment_weight
        self.constraint_weight = constraint_weight
        self.diversity_weight = diversity_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Load reference condition data
        self.conditions = RespiratoryConditions.get_all_conditions()
        self.condition_codes = list(self.conditions.keys())
    
    def compute_alignment_loss(
        self,
        sampled_symptoms: List[List[str]],
        condition_codes: List[str],
        bayesian_network
    ) -> torch.Tensor:
        """
        Compute alignment loss between sampled symptoms and expected patterns.
        
        Args:
            sampled_symptoms: List of symptom lists for each sample
            condition_codes: Corresponding condition codes
            bayesian_network: The learnable Bayesian network
            
        Returns:
            Alignment loss tensor
        """
        total_loss = 0.0
        num_samples = len(sampled_symptoms)
        
        for symptoms, code in zip(sampled_symptoms, condition_codes):
            if code not in self.conditions:
                continue
            
            condition_data = self.conditions[code]
            expected_primary = set(condition_data["symptoms"])
            expected_severity = set(condition_data["severity_indicators"])
            expected_all = expected_primary | expected_severity
            
            # Count how many sampled symptoms are expected
            sampled_set = set(symptoms)
            correct_symptoms = len(sampled_set & expected_all)
            incorrect_symptoms = len(sampled_set - expected_all)
            
            # Compute alignment score (higher is better)
            alignment_score = correct_symptoms / (len(sampled_set) + 1e-8)
            penalty = incorrect_symptoms / (len(sampled_set) + 1e-8)
            
            # Loss is negative alignment plus penalty
            total_loss += (1.0 - alignment_score) + penalty
        
        alignment_loss = total_loss / (num_samples + 1e-8)
        return torch.tensor(alignment_loss, device=self.device, dtype=torch.float32)
    
    def compute_constraint_loss(self, bayesian_network) -> torch.Tensor:
        """
        Compute constraint loss to keep probabilities in reasonable ranges.
        
        Args:
            bayesian_network: The learnable Bayesian network
            
        Returns:
            Constraint loss tensor
        """
        # Get current probability parameters
        primary_prob = torch.sigmoid(bayesian_network.primary_symptom_logit)
        severity_prob = torch.sigmoid(bayesian_network.severity_symptom_logit)
        
        # Primary symptoms should be likely (0.6 to 0.95)
        primary_target = 0.8
        primary_loss = F.mse_loss(primary_prob, torch.tensor(primary_target, device=self.device))
        
        # Severity symptoms should be less likely (0.2 to 0.6)
        severity_target = 0.4
        severity_loss = F.mse_loss(severity_prob, torch.tensor(severity_target, device=self.device))
        
        # Constraint on condition-specific weights (should stay near 0)
        if bayesian_network.condition_weights is not None:
            weight_regularization = torch.mean(bayesian_network.condition_weights ** 2)
        else:
            weight_regularization = torch.tensor(0.0, device=self.device)
        
        total_constraint_loss = primary_loss + severity_loss + 0.1 * weight_regularization
        return total_constraint_loss
    
    def compute_diversity_loss(self, bayesian_network) -> torch.Tensor:
        """
        Compute diversity loss to prevent all conditions from having identical distributions.
        
        Args:
            bayesian_network: The learnable Bayesian network
            
        Returns:
            Diversity loss tensor
        """
        if bayesian_network.condition_weights is None:
            return torch.tensor(0.0, device=self.device)
        
        # Encourage variation in condition-specific weights
        weights = bayesian_network.condition_weights
        
        # Compute variance across conditions for each weight dimension
        variance = torch.var(weights, dim=0)
        
        # We want high variance (diverse conditions), so loss is negative variance
        # Use negative log to handle small variances
        diversity_loss = -torch.log(variance.mean() + 1e-8)
        
        return diversity_loss
    
    def forward(
        self,
        sampled_symptoms: List[List[str]],
        condition_codes: List[str],
        bayesian_network
    ) -> torch.Tensor:
        """
        Compute total medical accuracy loss.
        
        Args:
            sampled_symptoms: List of symptom lists for each sample
            condition_codes: Corresponding condition codes
            bayesian_network: The learnable Bayesian network
            
        Returns:
            Total loss tensor
        """
        # Compute individual loss components
        alignment_loss = self.compute_alignment_loss(
            sampled_symptoms, condition_codes, bayesian_network
        )
        constraint_loss = self.compute_constraint_loss(bayesian_network)
        diversity_loss = self.compute_diversity_loss(bayesian_network)
        
        # Combine losses with weights
        total_loss = (
            self.alignment_weight * alignment_loss +
            self.constraint_weight * constraint_loss +
            self.diversity_weight * diversity_loss
        )
        
        return total_loss
    
    def get_loss_components(
        self,
        sampled_symptoms: List[List[str]],
        condition_codes: List[str],
        bayesian_network
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of loss components for logging.
        
        Args:
            sampled_symptoms: List of symptom lists for each sample
            condition_codes: Corresponding condition codes
            bayesian_network: The learnable Bayesian network
            
        Returns:
            Dictionary with loss components
        """
        alignment_loss = self.compute_alignment_loss(
            sampled_symptoms, condition_codes, bayesian_network
        )
        constraint_loss = self.compute_constraint_loss(bayesian_network)
        diversity_loss = self.compute_diversity_loss(bayesian_network)
        
        return {
            'alignment_loss': alignment_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'total_medical_loss': (
                self.alignment_weight * alignment_loss +
                self.constraint_weight * constraint_loss +
                self.diversity_weight * diversity_loss
            ).item()
        }
