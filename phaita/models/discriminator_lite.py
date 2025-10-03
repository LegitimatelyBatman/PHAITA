"""Lightweight discriminator for CPU-only or resource-constrained environments.

Uses DistilBERT (66M params, 2x faster than DeBERTa) with MLP-based graph module.
Vocabulary-based features as fallback when transformers unavailable.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from ..data.icd_conditions import RespiratoryConditions

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class VocabularyFeatureExtractor:
    """TF-IDF style feature extraction using symptom vocabulary.
    
    Fallback when transformers unavailable. Counts symptom keyword matches
    per condition to create sparse feature vectors.
    """
    
    def __init__(self, conditions: Dict):
        """Initialize with condition vocabulary."""
        self.conditions = conditions
        self.symptom_vocab = self._build_symptom_vocabulary()
        
    def _build_symptom_vocabulary(self) -> Dict[str, List[str]]:
        """Build vocabulary of symptoms per condition."""
        vocab = {}
        for code, data in self.conditions.items():
            symptoms = data['symptoms'] + data['severity_indicators'] + data['lay_terms']
            # Flatten to word-level tokens
            vocab[code] = [word.lower() for symptom in symptoms for word in symptom.split('_')]
        return vocab
    
    def extract_features(self, texts: List[str]) -> torch.Tensor:
        """Extract vocabulary-based features from texts.
        
        Returns: [batch_size, num_conditions] tensor with match counts
        """
        features = []
        for text in texts:
            text_lower = text.lower()
            text_words = set(text_lower.split())
            
            condition_features = []
            for code in sorted(self.conditions.keys()):
                # Count matching vocabulary words
                symptom_words = set(self.symptom_vocab[code])
                match_count = len(text_words & symptom_words)
                condition_features.append(float(match_count))
            
            features.append(condition_features)
        
        return torch.tensor(features)


class LightweightDiscriminator(nn.Module):
    """Lightweight discriminator using DistilBERT + MLP.
    
    Approximately 66M parameters (vs 140M for DeBERTa).
    2-3x faster inference, <1GB memory usage.
    """
    
    def __init__(
        self,
        use_pretrained: bool = True,
        freeze_encoder: bool = False,
        use_vocabulary_fallback: bool = True
    ):
        """Initialize lightweight discriminator.
        
        Args:
            use_pretrained: Load pretrained DistilBERT weights
            freeze_encoder: Freeze encoder weights for faster training
            use_vocabulary_fallback: Use vocabulary features if transformers unavailable
        """
        super().__init__()
        
        self.conditions = RespiratoryConditions.get_all_conditions()
        self.condition_codes = sorted(self.conditions.keys())
        self.num_conditions = len(self.condition_codes)
        
        # Initialize encoder
        if HAS_TRANSFORMERS and use_pretrained:
            self.encoder_name = 'distilbert-base-uncased'
            self.encoder = AutoModel.from_pretrained(self.encoder_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
            self.hidden_size = 768  # DistilBERT hidden size
            self.use_transformers = True
            
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            # Fallback to vocabulary features
            self.vocab_extractor = VocabularyFeatureExtractor(self.conditions)
            self.hidden_size = self.num_conditions  # Feature size = num conditions
            self.use_transformers = False
        
        # Simple MLP classifier (no GNN)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.num_conditions)
        )
        
        # Discriminator head (real vs synthetic)
        self.discriminator_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, texts: List[str], return_features: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass through discriminator.
        
        Args:
            texts: List of patient complaint strings
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with keys:
                - 'diagnosis_logits': [batch, num_conditions] logits
                - 'discriminator_scores': [batch, 1] real/fake scores
                - 'text_features': [batch, hidden_size] (if return_features=True)
        """
        if self.use_transformers:
            # Tokenize and encode with DistilBERT
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.encoder(**encoded)
            # Use [CLS] token representation
            text_features = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        else:
            # Use vocabulary features
            text_features = self.vocab_extractor.extract_features(texts).to(self.device)
        
        # Classification
        diagnosis_logits = self.classifier(text_features)
        discriminator_scores = self.discriminator_head(text_features)
        
        result = {
            'diagnosis_logits': diagnosis_logits,
            'discriminator_scores': discriminator_scores
        }
        
        if return_features:
            result['text_features'] = text_features
        
        return result
    
    def predict_diagnosis(self, texts: List[str], top_k: int = 3) -> List[List[Dict]]:
        """Predict top-k diagnoses for each text.
        
        Args:
            texts: List of patient complaints
            top_k: Number of top predictions to return
            
        Returns:
            List of lists, one per input. Each inner list contains dicts:
                - 'condition_code': ICD-10 code
                - 'condition_name': Human-readable name
                - 'probability': Softmax probability
        """
        self.eval()
        with torch.no_grad():
            outputs = self(texts)
            probs = torch.softmax(outputs['diagnosis_logits'], dim=-1)
        
        results = []
        for prob_dist in probs:
            top_probs, top_indices = torch.topk(prob_dist, k=min(top_k, len(prob_dist)))
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                code = self.condition_codes[idx]
                predictions.append({
                    'condition_code': code,
                    'condition_name': self.conditions[code]['name'],
                    'probability': prob.item()
                })
            results.append(predictions)
        
        return results
