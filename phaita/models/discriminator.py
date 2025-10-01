"""
Diagnosis discriminator using rule-based and keyword matching.
In production, this would use DeBERTa + GNN, but this is a mock implementation.
"""

import random
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from ..data.icd_conditions import RespiratoryConditions


class DiagnosisDiscriminator:
    """
    Discriminator for diagnosing conditions from patient complaints.
    Mock implementation using keyword matching.
    """
    
    def __init__(self):
        """Initialize the discriminator."""
        self.conditions = RespiratoryConditions.get_all_conditions()
        self._build_keyword_index()
        self.device = "cpu"  # Default device
        # Create a dummy parameter for optimizer compatibility
        self._dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))
    
    def _build_keyword_index(self) -> None:
        """Build keyword index for matching."""
        self.keyword_index = {}
        
        for code, data in self.conditions.items():
            keywords = set()
            
            # Add symptoms as keywords
            for symptom in data["symptoms"] + data["severity_indicators"]:
                keywords.add(symptom.replace('_', ' ').lower())
                # Add partial keywords
                for word in symptom.split('_'):
                    keywords.add(word.lower())
            
            # Add lay terms
            for term in data["lay_terms"]:
                keywords.add(term.lower())
                # Add individual words
                for word in term.split():
                    if len(word) > 3:  # Skip short words
                        keywords.add(word.lower())
            
            # Add condition name keywords
            for word in data["name"].lower().split():
                keywords.add(word.lower())
            
            self.keyword_index[code] = keywords
    
    def predict_diagnosis(
        self,
        complaints: List[str],
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Predict diagnosis from patient complaints.
        
        Args:
            complaints: List of patient complaint strings
            top_k: Number of top predictions to return per complaint
            
        Returns:
            List of (condition_code, confidence) tuples for each complaint
        """
        predictions = []
        
        for complaint in complaints:
            complaint_lower = complaint.lower()
            
            # Score each condition
            scores = {}
            for code, keywords in self.keyword_index.items():
                score = 0
                matches = 0
                
                for keyword in keywords:
                    if keyword in complaint_lower:
                        matches += 1
                        # Weight longer keywords more
                        score += len(keyword) / 10.0
                
                if matches > 0:
                    scores[code] = score
            
            # Get top predictions
            if scores:
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                # Normalize scores to confidences
                max_score = sorted_scores[0][1] if sorted_scores else 1.0
                predictions.append((
                    sorted_scores[0][0],
                    min(0.95, sorted_scores[0][1] / max_score)
                ))
            else:
                # Random guess if no matches
                predictions.append((
                    random.choice(list(self.conditions.keys())),
                    0.1
                ))
        
        return predictions
    
    def predict_with_explanation(
        self,
        complaint: str
    ) -> Tuple[str, float, Dict[str, any]]:
        """
        Predict diagnosis with explanation.
        
        Args:
            complaint: Patient complaint string
            
        Returns:
            Tuple of (condition_code, confidence, explanation_dict)
        """
        prediction = self.predict_diagnosis([complaint])[0]
        code, confidence = prediction
        
        # Build explanation
        complaint_lower = complaint.lower()
        matched_keywords = []
        
        for keyword in self.keyword_index[code]:
            if keyword in complaint_lower:
                matched_keywords.append(keyword)
        
        explanation = {
            "condition_code": code,
            "condition_name": self.conditions[code]["name"],
            "confidence": confidence,
            "matched_keywords": matched_keywords[:5],  # Top 5 matches
            "reasoning": f"Matched {len(matched_keywords)} symptoms/keywords"
        }
        
        return code, confidence, explanation
    
    def evaluate_batch(
        self,
        complaints: List[str],
        true_codes: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate discriminator on a batch of complaints.
        
        Args:
            complaints: List of complaints
            true_codes: List of true condition codes
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict_diagnosis(complaints)
        
        correct = 0
        total_confidence = 0.0
        
        for (pred_code, confidence), true_code in zip(predictions, true_codes):
            if pred_code == true_code:
                correct += 1
            total_confidence += confidence
        
        accuracy = correct / len(complaints) if complaints else 0.0
        avg_confidence = total_confidence / len(complaints) if complaints else 0.0
        
        return {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "correct": correct,
            "total": len(complaints)
        }
    
    def __call__(self, complaints: List[str], return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        PyTorch-compatible forward pass for adversarial training.
        
        Args:
            complaints: List of patient complaint strings
            return_features: Whether to return text features
            
        Returns:
            Dictionary with:
                - diagnosis_logits: [batch_size, num_conditions]
                - discriminator_scores: [batch_size, 1] 
                - text_features: [batch_size, 768] (if return_features=True)
        """
        batch_size = len(complaints)
        num_conditions = len(self.conditions)
        
        # Create mock diagnosis logits (include dummy param to enable gradients)
        diagnosis_logits = torch.zeros(batch_size, num_conditions, device=self.device, requires_grad=True)
        diagnosis_logits = diagnosis_logits + self._dummy_param * 0  # Include dummy param in computation
        
        # Get predictions and convert to logits
        predictions = self.predict_diagnosis(complaints)
        with torch.no_grad():
            for i, (pred_code, confidence) in enumerate(predictions):
                # Find index of predicted condition
                condition_codes = list(self.conditions.keys())
                if pred_code in condition_codes:
                    pred_idx = condition_codes.index(pred_code)
                    diagnosis_logits.data[i, pred_idx] = confidence * 10  # Scale confidence to logit-like values
        
        # Mock discriminator scores (real vs fake) - include dummy param
        discriminator_scores = torch.rand(batch_size, 1, device=self.device, requires_grad=True) * 0.5 + 0.5
        discriminator_scores = discriminator_scores + self._dummy_param * 0
        
        result = {
            "diagnosis_logits": diagnosis_logits,
            "discriminator_scores": discriminator_scores
        }
        
        # Add text features if requested
        if return_features:
            # Mock text features - include dummy param
            text_features = torch.randn(batch_size, 768, device=self.device, requires_grad=True)
            text_features = text_features + self._dummy_param * 0
            result["text_features"] = text_features
        
        return result
    
    def to(self, device):
        """
        Move discriminator to specified device (PyTorch compatibility).
        
        Args:
            device: Device to move to ('cpu' or 'cuda')
            
        Returns:
            Self for method chaining
        """
        self.device = device if isinstance(device, str) else str(device)
        return self
    
    def train(self, mode: bool = True):
        """
        Set discriminator to training mode (PyTorch compatibility).
        
        Args:
            mode: Whether to set to training mode
        """
        # No-op for mock implementation
        pass
    
    def eval(self):
        """Set discriminator to evaluation mode (PyTorch compatibility)."""
        # No-op for mock implementation
        pass
    
    def parameters(self):
        """
        Return parameters for optimizer (PyTorch compatibility).
        
        Returns:
            List with dummy parameter for optimizer compatibility
        """
        return [self._dummy_param]
    
    def state_dict(self) -> Dict:
        """
        Return state dictionary for checkpointing.
        
        Returns:
            Dictionary with model state
        """
        return {
            "keyword_index": self.keyword_index,
            "device": self.device
        }
    
    def load_state_dict(self, state_dict: Dict):
        """
        Load state from dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        if "keyword_index" in state_dict:
            self.keyword_index = state_dict["keyword_index"]
        if "device" in state_dict:
            self.device = state_dict["device"]
