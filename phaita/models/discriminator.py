"""Diagnosis discriminator using DeBERTa encoder and Graph Neural Networks.

Combines text features from patient complaints with symptom graph relationships
and exposes convenience helpers for ranked differential diagnosis generation.
When ``use_pretrained`` is ``False`` the model falls back to a lightweight
keyword-based encoder that does not require transformers or torch-geometric.
"""

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.icd_conditions import RespiratoryConditions
from ..utils.model_loader import load_model_and_tokenizer, ModelDownloadError
from .discriminator_lite import VocabularyFeatureExtractor

# Optional dependencies are imported lazily to avoid heavy requirements when
# using the lightweight fallback path. SymptomGraphModule will be loaded inside
# the constructor only when requested.
SymptomGraphModule = None


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
        dropout: float = 0.1,
        use_causal_edges: bool = True
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
            use_causal_edges: Whether to use causal edges in GNN (default True)
        
        Raises:
            ValueError: If use_pretrained is False
            RuntimeError: If model loading fails
        """
        super().__init__()

        self.use_pretrained = use_pretrained
        self.vocab_extractor: Optional[VocabularyFeatureExtractor] = None
        self.tokenizer = None
        self.text_encoder = None
        self.gnn = None
        self.graph_feature_dim = 0

        self.conditions = RespiratoryConditions.get_all_conditions()
        self.num_conditions = len(self.conditions)
        self.condition_codes = list(self.conditions.keys())
        self.keyword_index = self._build_keyword_index(self.conditions)

        RespiratoryConditions.register_reload_hook(self.reload_conditions)

        if self.use_pretrained:
            try:
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "transformers is required when use_pretrained=True. "
                    "Install with: pip install transformers==4.46.0\n"
                    "GPU Requirements: CUDA-capable GPU with 4GB+ VRAM recommended for full functionality. "
                    "CPU-only mode available but slower."
                ) from e

            # Initialize text encoder (DeBERTa) with retry logic
            try:
                self.text_encoder, self.tokenizer = load_model_and_tokenizer(
                    model_name=model_name,
                    model_type="auto",
                    max_retries=3,
                    timeout=300
                )
                self.text_feature_dim = self.text_encoder.config.hidden_size  # 768 for base

                if freeze_encoder:
                    for param in self.text_encoder.parameters():
                        param.requires_grad = False
            except ModelDownloadError as e:
                raise RuntimeError(
                    f"Failed to load text encoder {model_name}. "
                    f"{e}\n"
                    f"Requirements:\n"
                    f"- transformers==4.46.0\n"
                    f"- torch==2.5.1\n"
                    f"- Internet connection to download model from HuggingFace Hub"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load text encoder {model_name}. "
                    f"Error: {e}\n"
                    f"Requirements:\n"
                    f"- transformers==4.46.0\n"
                    f"- torch==2.5.1\n"
                    f"- Internet connection to download model from HuggingFace Hub"
                ) from e

            # Initialize GNN for symptom relationships lazily
            global SymptomGraphModule
            if SymptomGraphModule is None:
                try:
                    from .gnn_module import SymptomGraphModule as _SymptomGraphModule  # type: ignore
                    SymptomGraphModule = _SymptomGraphModule
                except ImportError as e:
                    raise ImportError(
                        "torch-geometric is required when use_pretrained=True to construct the GNN module. "
                        "Install with: pip install torch-geometric==2.6.1"
                    ) from e

            try:
                assert SymptomGraphModule is not None  # for type checkers
                self.gnn = SymptomGraphModule(
                    conditions=self.conditions,
                    hidden_dim=gnn_hidden_dim,
                    output_dim=gnn_output_dim,
                    dropout=dropout,
                    use_causal_edges=use_causal_edges
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
        else:
            # Lightweight keyword-matching encoder fallback. Provides a
            # deterministic feature representation that keeps unit tests
            # dependency-free while preserving downstream behaviour.
            self.vocab_extractor = VocabularyFeatureExtractor(self.conditions)
            self.text_feature_dim = self.num_conditions
            self.graph_feature_dim = 0

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

    @staticmethod
    def _generate_keyword_variants(term: str) -> List[str]:
        """Generate normalized keyword variants for matching free-text complaints."""

        normalized = term.strip().lower()
        if not normalized:
            return []

        variants = [normalized]
        underscored = normalized.replace("_", " ")
        if underscored != normalized:
            variants.append(underscored)
        return variants

    def _build_keyword_index(self, conditions: Optional[Dict[str, Dict]] = None) -> Dict[str, List[str]]:
        """Construct lookup of canonical keywords for quick explanation matches."""

        if conditions is None:
            conditions = self.conditions

        keyword_index: Dict[str, List[str]] = {}
        for code, data in conditions.items():
            seen: Set[str] = set()
            keywords: List[str] = []

            for field in ("symptoms", "severity_indicators", "lay_terms"):
                for term in data.get(field, []):
                    for variant in self._generate_keyword_variants(term):
                        if variant not in seen:
                            seen.add(variant)
                            keywords.append(variant)

            demographics = data.get("demographics", {})
            history = data.get("history", {})

            for term in self._flatten_demographic_terms(demographics):
                if term not in seen:
                    seen.add(term)
                    keywords.append(term)
            for term in self._flatten_history_terms(history):
                if term not in seen:
                    seen.add(term)
                    keywords.append(term)

            keyword_index[code] = keywords

        return keyword_index

    @staticmethod
    def _flatten_demographic_terms(demographics: Dict[str, Any]) -> List[str]:
        terms: List[str] = []
        for block in demographics.values():
            age_ranges = block.get("age_ranges") if isinstance(block, dict) else None
            if age_ranges:
                for entry in age_ranges:
                    minimum = entry.get("min")
                    maximum = entry.get("max")
                    if minimum is None and maximum is None:
                        continue
                    if minimum == maximum:
                        terms.append(f"age {int(minimum)}")
                    else:
                        terms.append(f"age {int(minimum)}-{int(maximum)}")
            for key, value in block.items():
                if key == "age_ranges":
                    continue
                if isinstance(value, list):
                    for entry in value:
                        normalized = str(entry).strip().lower()
                        if normalized:
                            terms.append(normalized)
        return terms

    @staticmethod
    def _flatten_history_terms(history: Dict[str, Any]) -> List[str]:
        terms: List[str] = []
        for block in history.values():
            if not isinstance(block, dict):
                continue
            for key, value in block.items():
                if isinstance(value, list):
                    for entry in value:
                        normalized = str(entry).strip().lower()
                        if normalized:
                            terms.append(normalized)
                elif key == "last_meal" and value:
                    for entry in value:
                        normalized = str(entry).strip().lower()
                        if normalized:
                            terms.append(normalized)
        return terms

    def reload_conditions(self, conditions: Optional[Dict[str, Dict]] = None) -> None:
        """Reload respiratory conditions and rebuild keyword index."""

        if conditions is None:
            conditions = RespiratoryConditions.get_all_conditions()

        self.conditions = conditions
        self.condition_codes = list(self.conditions.keys())
        self.num_conditions = len(self.conditions)
        self.keyword_index = self._build_keyword_index(self.conditions)
            
    def encode_text(self, complaints: List[str]) -> torch.Tensor:
        """
        Encode text complaints to feature vectors using DeBERTa.

        Args:
            complaints: List of patient complaint strings

        Returns:
            Text features [batch_size, text_feature_dim]
        """
        device = next(self.parameters()).device

        if not self.use_pretrained:
            assert self.vocab_extractor is not None
            features = self.vocab_extractor.extract_features(complaints).to(device)
            return features.float()

        try:
            if self.tokenizer is None or self.text_encoder is None:
                raise RuntimeError("Pretrained encoder not initialised.")

            inputs = self.tokenizer(
                complaints,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}

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
        device = next(self.parameters()).device

        if self.gnn is None or self.graph_feature_dim == 0:
            return torch.zeros(batch_size, self.graph_feature_dim, device=device)

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
        top_k: int = 1,
        use_mc_dropout: bool = True,
        num_mc_samples: int = 20
    ) -> List[List[Dict[str, Any]]]:
        """Predict ranked differential diagnosis lists for patient complaints.

        Args:
            complaints: List of patient complaint strings.
            top_k: Number of top predictions to return per complaint. The
                predictions are ordered by probability (highest first).
            use_mc_dropout: Whether to use Monte Carlo Dropout for uncertainty
                estimation (default: True). When False, uses single forward pass.
            num_mc_samples: Number of MC dropout samples for uncertainty estimation
                (default: 20). Only used if use_mc_dropout is True.

        Returns:
            A list with one entry per complaint. Each entry is a list of
            dictionaries containing:
                - ``condition_code``: ICD-10 code of the diagnosis
                - ``condition_name``: Friendly condition name
                - ``probability``: Model probability for the diagnosis
                - ``confidence_interval``: Tuple of (lower, upper) bounds
                  computed via a Wilson score interval when MC dropout
                  sampling is available. Falls back to a degenerate interval
                  around the point estimate when no sampling is performed.
                - ``evidence``: Supporting symptom and severity indicators
                - ``uncertainty``: Uncertainty score (0-1, lower is more certain)
                - ``confidence_level``: "high" or "low" based on uncertainty threshold
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Get predictions with or without MC dropout
        if use_mc_dropout:
            mean_probs, std_probs = self._mc_dropout_sample(complaints, num_mc_samples)
            # Calculate entropy for epistemic uncertainty
            entropy_scores = self._calculate_entropy(mean_probs)
        else:
            self.eval()
            with torch.no_grad():
                outputs = self.forward(complaints)
                diagnosis_logits = outputs["diagnosis_logits"]
                mean_probs = F.softmax(diagnosis_logits, dim=1)
                std_probs = torch.zeros_like(mean_probs)
                entropy_scores = self._calculate_entropy(mean_probs)

        ranked_predictions: List[List[Dict[str, Any]]] = []
        
        for idx, prob_distribution in enumerate(mean_probs):
            complaint_predictions: List[Dict[str, Any]] = []
            
            # Calculate uncertainty metrics for this complaint
            entropy = entropy_scores[idx].item()
            # Normalize entropy to [0, 1] range (max entropy is log(num_conditions))
            max_entropy = math.log(self.num_conditions)
            normalized_entropy = min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0
            
            # For MC dropout, also consider prediction variance
            if use_mc_dropout:
                prediction_variance = std_probs[idx].mean().item()
                # Combine entropy and variance for overall uncertainty
                uncertainty_score = 0.7 * normalized_entropy + 0.3 * min(prediction_variance * 10, 1.0)
            else:
                uncertainty_score = normalized_entropy
            
            # Extract the ordered probabilities for this complaint
            k = min(top_k, prob_distribution.shape[0])
            top_probs, top_indices = torch.topk(prob_distribution, k=k)

            for probability, condition_idx in zip(top_probs, top_indices):
                prob_value = probability.item()
                condition_code = self.condition_codes[condition_idx.item()]
                condition_data = RespiratoryConditions.get_condition_by_code(condition_code)
                
                # Estimate a credible interval for the probability. When MC dropout
                # sampling is enabled we use the true number of samples collected.
                # Without sampling (deterministic forward pass) the helper returns a
                # degenerate interval anchored at the point estimate.
                sample_count = num_mc_samples if use_mc_dropout else 0
                confidence_interval = self._estimate_confidence_interval(prob_value, sample_count)

                evidence = {
                    "key_symptoms": condition_data.get("symptoms", [])[:3],
                    "severity_indicators": condition_data.get("severity_indicators", [])[:2],
                    "description": condition_data.get("description", "")
                }

                # Determine confidence level based on uncertainty threshold
                confidence_level = "high" if uncertainty_score < 0.3 else "low"

                complaint_predictions.append({
                    "condition_code": condition_code,
                    "condition_name": condition_data.get("name", condition_code),
                    "probability": prob_value,
                    "confidence_interval": confidence_interval,
                    "evidence": evidence,
                    "uncertainty": uncertainty_score,
                    "confidence_level": confidence_level
                })

            ranked_predictions.append(complaint_predictions)

        return ranked_predictions

    @staticmethod
    def _estimate_confidence_interval(probability: float, sample_size: int) -> tuple:
        """Compute a Wilson score 95% interval for a Bernoulli proportion.

        Args:
            probability: Point estimate of the success probability.
            sample_size: Number of independent Monte Carlo samples that
                produced the estimate. When zero (no sampling), the method
                returns a degenerate interval anchored at the point estimate.

        Returns:
            Tuple of ``(lower, upper)`` bounds clipped to ``[0, 1]``.
        """
        clamped_prob = min(max(probability, 0.0), 1.0)

        if sample_size <= 0:
            return (clamped_prob, clamped_prob)

        z = 1.96  # 95% confidence level
        n = float(sample_size)
        z_sq = z ** 2
        denominator = 1.0 + z_sq / n
        centre = clamped_prob + z_sq / (2.0 * n)
        margin_term = math.sqrt((clamped_prob * (1.0 - clamped_prob) / n) + (z_sq / (4.0 * n ** 2)))
        lower = (centre - z * margin_term) / denominator
        upper = (centre + z * margin_term) / denominator

        return (max(0.0, lower), min(1.0, upper))
    
    @staticmethod
    def _calculate_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of probability distribution.
        
        Entropy measures uncertainty in the prediction:
        - High entropy = uncertain prediction (probabilities spread out)
        - Low entropy = confident prediction (probabilities concentrated)
        
        Args:
            probabilities: Probability distribution tensor [batch_size, num_classes]
            
        Returns:
            Entropy values [batch_size]
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        log_probs = torch.log(probabilities + epsilon)
        entropy = -torch.sum(probabilities * log_probs, dim=1)
        return entropy
    
    def _mc_dropout_sample(
        self,
        complaints: List[str],
        num_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal method: Predict with uncertainty using Monte Carlo Dropout.
        
        This method enables dropout during inference and runs multiple forward
        passes to estimate model uncertainty (epistemic uncertainty). The variance
        in predictions across samples indicates how uncertain the model is.
        
        This is an internal method that returns raw tensors. For a user-friendly
        API, use predict_with_uncertainty() instead.
        
        Args:
            complaints: List of patient complaint strings
            num_samples: Number of MC dropout samples to collect (default: 20)
            
        Returns:
            Tuple of (mean_probabilities, std_probabilities):
                - mean_probabilities: Mean predictions [batch_size, num_conditions]
                - std_probabilities: Standard deviation [batch_size, num_conditions]
        """
        # Enable dropout for uncertainty estimation
        self.train()
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.forward(complaints)
                diagnosis_logits = outputs["diagnosis_logits"]
                probs = F.softmax(diagnosis_logits, dim=1)
                predictions.append(probs)
        
        # Stack predictions and compute statistics
        predictions_stack = torch.stack(predictions, dim=0)  # [num_samples, batch_size, num_conditions]
        mean_probs = predictions_stack.mean(dim=0)
        std_probs = predictions_stack.std(dim=0)
        
        # Return to eval mode
        self.eval()
        
        return mean_probs, std_probs
    
    def predict_with_uncertainty(
        self,
        complaints: List[str],
        num_samples: int = 20,
        top_k: int = 3
    ) -> List[List[Dict[str, float]]]:
        """Predict diagnosis with epistemic uncertainty using Monte Carlo Dropout.
        
        Performs multiple forward passes with dropout enabled to estimate uncertainty.
        Higher uncertainty indicates the model is less confident in its prediction.
        
        Args:
            complaints: List of patient complaint strings
            num_samples: Number of MC samples to draw (default 20)
            top_k: Number of top predictions to return per complaint
            
        Returns:
            List of lists, one per complaint. Each inner list contains dicts with keys:
                - 'condition_code': ICD-10 code
                - 'condition_name': Human-readable name
                - 'probability': Mean probability across MC samples
                - 'uncertainty': Standard deviation across MC samples
                - 'confidence_level': 'high' (<0.3), 'medium' (0.3-0.6), 'low' (>0.6)
                
        Example:
            >>> disc = DiagnosisDiscriminator()
            >>> results = disc.predict_with_uncertainty(["I can't breathe"])
            >>> print(results[0][0])
            {
                'condition_code': 'J45.9',
                'condition_name': 'Asthma',
                'probability': 0.75,
                'uncertainty': 0.12,
                'confidence_level': 'high'
            }
        """
        # Enable dropout for uncertainty estimation
        was_training = self.training
        self.train()  # Enable dropout layers
        
        all_predictions = []
        
        # Perform multiple forward passes with dropout
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self(complaints, return_features=False)
                probs = F.softmax(outputs['diagnosis_logits'], dim=-1)
                all_predictions.append(probs)
        
        # Restore original training state
        if not was_training:
            self.eval()
        
        # Compute statistics across MC samples
        predictions_tensor = torch.stack(all_predictions)  # [num_samples, batch, num_conditions]
        mean_probs = predictions_tensor.mean(dim=0)  # [batch, num_conditions]
        std_probs = predictions_tensor.std(dim=0)    # [batch, num_conditions]
        
        # Convert to list of results
        batch_results = []
        for batch_idx in range(len(complaints)):
            # Get all conditions with their stats
            condition_results = []
            for condition_idx, code in enumerate(self.condition_codes):
                prob = mean_probs[batch_idx, condition_idx].item()
                uncertainty = std_probs[batch_idx, condition_idx].item()
                
                # Classify confidence level based on uncertainty
                if uncertainty < 0.3:
                    confidence_level = 'high'
                elif uncertainty < 0.6:
                    confidence_level = 'medium'
                else:
                    confidence_level = 'low'
                
                condition_results.append({
                    'condition_code': code,
                    'condition_name': self.conditions[code]['name'],
                    'probability': prob,
                    'uncertainty': uncertainty,
                    'confidence_level': confidence_level
                })
            
            # Sort by probability and take top_k
            condition_results.sort(key=lambda x: x['probability'], reverse=True)
            batch_results.append(condition_results[:top_k])
        
        return batch_results
    
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
        else:
            self.keyword_index = self._build_keyword_index()
        if "condition_codes" in state_dict:
            self.condition_codes = state_dict.pop("condition_codes")
        
        # Load model parameters
        super().load_state_dict(state_dict, strict=strict)
