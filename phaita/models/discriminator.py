"""Diagnosis discriminator using DeBERTa encoder and Graph Neural Networks.

Combines text features from patient complaints with symptom graph relationships
and exposes convenience helpers for ranked differential diagnosis generation.
Requires transformers and torch-geometric to be properly installed.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.icd_conditions import RespiratoryConditions

# Enforce required dependencies
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    raise ImportError(
        "transformers is required for DiagnosisDiscriminator. "
        "Install with: pip install transformers==4.46.0\n"
        "GPU Requirements: CUDA-capable GPU with 4GB+ VRAM recommended for full functionality. "
        "CPU-only mode available but slower."
    ) from e

try:
    from .gnn_module import SymptomGraphModule
except ImportError as e:
    raise ImportError(
        "GNN module is required for DiagnosisDiscriminator. "
        "Ensure torch-geometric==2.6.1 is properly installed."
    ) from e


class DiagnosisDiscriminator(nn.Module):
    """
    Discriminator for diagnosing conditions from patient complaints.
    Uses DeBERTa for text encoding and GAT for symptom relationship modeling.
    Requires transformers and torch-geometric to be installed.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        use_pretrained: bool = True,
        freeze_encoder: bool = False,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 256,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize the discriminator.
        
        Args:
            model_name: Name of the DeBERTa model to use
            use_pretrained: Whether to use pretrained weights (must be True)
            freeze_encoder: Whether to freeze the encoder weights
            gnn_hidden_dim: Hidden dimension for GNN
            gnn_output_dim: Output dimension for GNN
            fusion_hidden_dim: Hidden dimension for fusion layer
            dropout: Dropout rate
        
        Raises:
            ValueError: If use_pretrained is False
            RuntimeError: If model loading fails
        """
        super().__init__()
        
        if not use_pretrained:
            raise ValueError(
                "DiagnosisDiscriminator requires use_pretrained=True. "
                "Fallback encoder has been removed. "
                "Ensure transformers and torch-geometric are properly installed."
            )
        
        self.conditions = RespiratoryConditions.get_all_conditions()
        self.num_conditions = len(self.conditions)
        self.condition_codes = list(self.conditions.keys())
        
        # Initialize text encoder (DeBERTa)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = AutoModel.from_pretrained(model_name)
            self.text_feature_dim = self.text_encoder.config.hidden_size  # 768 for base
            
            if freeze_encoder:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
        except Exception as e:
            raise RuntimeError(
                f"Failed to load text encoder {model_name}. "
                f"Error: {e}\n"
                f"Requirements:\n"
                f"- transformers==4.46.0\n"
                f"- torch==2.5.1\n"
                f"- Internet connection to download model from HuggingFace Hub"
            ) from e
        
        # Initialize GNN for symptom relationships
        try:
            self.gnn = SymptomGraphModule(
                conditions=self.conditions,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_output_dim,
                dropout=dropout
            )
            self.graph_feature_dim = gnn_output_dim
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize GNN module. "
                f"Error: {e}\n"
                f"Requirements:\n"
                f"- torch-geometric==2.6.1\n"
                f"- torch==2.5.1"
            ) from e
        
        # Fusion layer (combines text + graph features)
        combined_dim = self.text_feature_dim + self.graph_feature_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head (diagnosis prediction)
        self.diagnosis_head = nn.Linear(fusion_hidden_dim // 2, self.num_conditions)
        
        # Discriminator head (real vs fake)
        self.discriminator_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
            
    def encode_text(self, complaints: List[str]) -> torch.Tensor:
        """
        Encode text complaints to feature vectors using DeBERTa.
        
        Args:
            complaints: List of patient complaint strings
            
        Returns:
            Text features [batch_size, text_feature_dim]
        """
        try:
            inputs = self.tokenizer(
                complaints,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to same device as model
            inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
            
            with torch.set_grad_enabled(self.training):
                outputs = self.text_encoder(**inputs)
                # Use [CLS] token representation
                text_features = outputs.last_hidden_state[:, 0, :]
            
            return text_features
        except Exception as e:
            raise RuntimeError(
                f"Error in DeBERTa encoding: {e}\n"
                f"Ensure model is properly loaded and device has sufficient memory."
            ) from e
    
    def get_graph_features(self, batch_size: int) -> torch.Tensor:
        """
        Get graph features for the batch.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Graph features [batch_size, graph_feature_dim]
        """
        return self.gnn(batch_size)
    
    def forward(
        self, 
        complaints: List[str], 
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the discriminator.
        
        Args:
            complaints: List of patient complaint strings
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with:
                - diagnosis_logits: [batch_size, num_conditions]
                - discriminator_scores: [batch_size, 1]
                - text_features: [batch_size, text_feature_dim] (if return_features=True)
                - graph_features: [batch_size, graph_feature_dim] (if return_features=True)
                - fused_features: [batch_size, fusion_dim] (if return_features=True)
        """
        batch_size = len(complaints)
        
        # Encode text with DeBERTa
        text_features = self.encode_text(complaints)
        
        # Get graph features
        graph_features = self.get_graph_features(batch_size)
        
        # Fuse text and graph features
        combined_features = torch.cat([text_features, graph_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # Diagnosis prediction
        diagnosis_logits = self.diagnosis_head(fused_features)
        
        # Real vs fake discrimination
        discriminator_scores = self.discriminator_head(fused_features)
        
        result = {
            "diagnosis_logits": diagnosis_logits,
            "discriminator_scores": discriminator_scores
        }
        
        if return_features:
            result["text_features"] = text_features
            result["graph_features"] = graph_features
            result["fused_features"] = fused_features
        
        return result
    
    def predict_diagnosis(
        self,
        complaints: List[str],
        top_k: int = 1
    ) -> List[List[Dict[str, Any]]]:
        """Predict ranked differential diagnosis lists for patient complaints.

        Args:
            complaints: List of patient complaint strings.
            top_k: Number of top predictions to return per complaint. The
                predictions are ordered by probability (highest first).

        Returns:
            A list with one entry per complaint. Each entry is a list of
            dictionaries containing:
                - ``condition_code``: ICD-10 code of the diagnosis
                - ``condition_name``: Friendly condition name
                - ``probability``: Model probability for the diagnosis
                - ``confidence_interval``: Tuple of (lower, upper) bounds
                - ``evidence``: Supporting symptom and severity indicators
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self.eval()
        ranked_predictions: List[List[Dict[str, Any]]] = []
        with torch.no_grad():
            outputs = self.forward(complaints)
            diagnosis_logits = outputs["diagnosis_logits"]

            # Convert logits to probabilities
            probs = F.softmax(diagnosis_logits, dim=1)

            for idx, prob_distribution in enumerate(probs):
                complaint_predictions: List[Dict[str, Any]] = []
                logits_for_complaint = diagnosis_logits[idx]
                logit_spread = (logits_for_complaint.max() - logits_for_complaint.min()).item()
                effective_sample_size = max(int(logit_spread * 10), 20)

                # Extract the ordered probabilities for this complaint
                k = min(top_k, prob_distribution.shape[0])
                top_probs, top_indices = torch.topk(prob_distribution, k=k)

                for probability, condition_idx in zip(top_probs, top_indices):
                    prob_value = probability.item()
                    condition_code = self.condition_codes[condition_idx.item()]
                    condition_data = RespiratoryConditions.get_condition_by_code(condition_code)
                    confidence_interval = self._estimate_confidence_interval(
                        prob_value,
                        effective_sample_size
                    )

                    evidence = {
                        "key_symptoms": condition_data.get("symptoms", [])[:3],
                        "severity_indicators": condition_data.get("severity_indicators", [])[:2],
                        "description": condition_data.get("description", "")
                    }

                    complaint_predictions.append({
                        "condition_code": condition_code,
                        "condition_name": condition_data.get("name", condition_code),
                        "probability": prob_value,
                        "confidence_interval": confidence_interval,
                        "evidence": evidence
                    })

                ranked_predictions.append(complaint_predictions)

        return ranked_predictions

    @staticmethod
    def _estimate_confidence_interval(probability: float, sample_size: int) -> tuple:
        """Estimate a simple 95% confidence interval around a probability.

        The interval is computed using a normal approximation of the
        binomial distribution with an empirically derived pseudo sample size.
        The bounds are clipped to the valid range of [0, 1].
        """
        pseudo_n = max(sample_size, 1)
        standard_error = math.sqrt(max(probability * (1.0 - probability) / pseudo_n, 1e-6))
        margin = 1.96 * standard_error
        lower = max(0.0, probability - margin)
        upper = min(1.0, probability + margin)
        return (lower, upper)
    
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
        self.eval()
        with torch.no_grad():
            outputs = self.forward([complaint], return_features=True)
            diagnosis_logits = outputs["diagnosis_logits"][0]
            
            # Get prediction
            probs = F.softmax(diagnosis_logits, dim=0)
            top_prob, top_idx = probs.max(dim=0)
            
            code = self.condition_codes[top_idx.item()]
            confidence = top_prob.item()
            
            # Build explanation with attention-like scores
            complaint_lower = complaint.lower()
            matched_keywords = []
            
            for keyword in self.keyword_index.get(code, []):
                if keyword in complaint_lower:
                    matched_keywords.append(keyword)
            
            explanation = {
                "condition_code": code,
                "condition_name": self.conditions[code]["name"],
                "confidence": confidence,
                "matched_keywords": matched_keywords[:5],
                "reasoning": f"Neural model prediction with {confidence:.2%} confidence",
                "top_3_predictions": [
                    (self.condition_codes[i], probs[i].item())
                    for i in probs.topk(min(3, len(self.condition_codes))).indices
                ]
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
        predictions = self.predict_diagnosis(complaints, top_k=1)

        correct = 0
        total_confidence = 0.0

        for ranked_predictions, true_code in zip(predictions, true_codes):
            top_prediction = ranked_predictions[0]
            pred_code = top_prediction["condition_code"]
            confidence = top_prediction["probability"]
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
        Alias for forward() method.
        
        Args:
            complaints: List of patient complaint strings
            return_features: Whether to return text features
            
        Returns:
            Dictionary with model outputs
        """
        return self.forward(complaints, return_features)
    
    def to(self, device):
        """
        Move discriminator to specified device (PyTorch compatibility).
        
        Args:
            device: Device to move to ('cpu' or 'cuda')
            
        Returns:
            Self for method chaining
        """
        return super().to(device)
    
    def state_dict(self) -> Dict:
        """
        Return state dictionary for checkpointing.
        Overrides nn.Module.state_dict() to include custom fields.
        
        Returns:
            Dictionary with model state
        """
        state = super().state_dict()
        # Add custom fields
        state['keyword_index'] = self.keyword_index
        state['condition_codes'] = self.condition_codes
        return state
    
    def load_state_dict(self, state_dict: Dict, strict: bool = True):
        """
        Load state from dictionary.
        
        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly enforce key matching
        """
        # Extract custom fields
        if "keyword_index" in state_dict:
            self.keyword_index = state_dict.pop("keyword_index")
        if "condition_codes" in state_dict:
            self.condition_codes = state_dict.pop("condition_codes")
        
        # Load model parameters
        super().load_state_dict(state_dict, strict=strict)
