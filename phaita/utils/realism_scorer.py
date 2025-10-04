"""
Realism scoring module for medical complaints.
Uses language models to assess the authenticity and realism of generated patient complaints.
Requires transformers to be properly installed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from .model_loader import robust_model_download, ModelDownloadError
from .dependency_versions import (
    TRANSFORMERS_VERSION,
    format_install_instruction,
    format_transformer_requirements,
)

# Enforce required dependencies
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    from transformers import AutoModelForCausalLM
except ImportError as e:
    raise ImportError(
        "transformers is required for RealismScorer. "
        f"{format_install_instruction('transformers', TRANSFORMERS_VERSION)}\n"
        "GPU Requirements: CUDA-capable GPU with 4GB+ VRAM recommended for full functionality. "
        "CPU-only mode available but slower."
    ) from e

class RealismScorer:
    """
    Assesses the realism of patient complaints using pre-trained language models.
    Requires transformers to be installed.
    """
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 use_medical_model: bool = True,
                 device: Optional[str] = None):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        try:
            if use_medical_model:
                medical_models = [
                    "emilyalsentzer/Bio_ClinicalBERT",
                    "dmis-lab/biobert-base-cased-v1.1",
                    "bert-base-uncased"  # Fallback
                ]

                last_error: Optional[Exception] = None
                for candidate in medical_models:
                    try:
                        self.model = robust_model_download(
                            model_name=candidate,
                            model_type="auto",
                            max_retries=3,
                            timeout=300
                        )
                        self.tokenizer = robust_model_download(
                            model_name=candidate,
                            model_type="tokenizer",
                            max_retries=3,
                            timeout=300
                        )
                        self.model = self.model.to(self.device)
                        self.logger.info(f"Loaded realism model: {candidate}")
                        break
                    except (ModelDownloadError, Exception) as load_error:
                        last_error = load_error
                        continue
                else:
                    # All preferred models failed - raise error
                    requirements = format_transformer_requirements(
                        internet_note="- Internet connection to download models from HuggingFace Hub"
                    )
                    raise RuntimeError(
                        f"Failed to load any preferred medical model. Last error: {last_error}\n"
                        f"Attempted models: {medical_models}\n"
                        f"{requirements}"
                    )
            else:
                self.tokenizer = robust_model_download(
                    model_name=model_name,
                    model_type="tokenizer",
                    max_retries=3,
                    timeout=300
                )
                self.model = robust_model_download(
                    model_name=model_name,
                    model_type="auto",
                    max_retries=3,
                    timeout=300
                )
                self.model = self.model.to(self.device)

            self.model.eval()

            # Initialize perplexity scorer
            try:
                self.perplexity_tokenizer = robust_model_download(
                    model_name="gpt2",
                    model_type="tokenizer",
                    max_retries=3,
                    timeout=300
                )
                # gpt2 has no pad token by default; align with eos token for batching
                if self.perplexity_tokenizer.pad_token is None:
                    self.perplexity_tokenizer.pad_token = self.perplexity_tokenizer.eos_token

                self.perplexity_model = robust_model_download(
                    model_name="gpt2",
                    model_type="causal_lm",
                    max_retries=3,
                    timeout=300
                )
                self.perplexity_model = self.perplexity_model.to(self.device)
                self.perplexity_model.eval()
            except (ModelDownloadError, Exception) as e:
                requirements = format_transformer_requirements(
                    internet_note="- Internet connection to download models from HuggingFace Hub"
                )
                raise RuntimeError(
                    f"Failed to initialize perplexity model (gpt2): {e}\n"
                    f"{requirements}"
                ) from e

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize realism scorer: {e}\n"
                f"Ensure all dependencies are properly installed."
            ) from e
    
    def compute_realism_score(self, complaints: List[str]) -> torch.Tensor:
        """
        Compute realism scores for a batch of complaints.
        
        Args:
            complaints: List of patient complaint texts
            
        Returns:
            Tensor of realism scores [0, 1] where higher is more realistic
        """
        if self.model is None:
            # Return mock scores if model not available
            return torch.tensor([0.8 + 0.1 * np.random.randn() for _ in complaints], 
                              device=self.device, dtype=torch.float32)
        
        scores = []
        
        for complaint in complaints:
            # Compute multiple realism metrics
            fluency_score = self._compute_fluency_score(complaint)
            coherence_score = self._compute_coherence_score(complaint)
            medical_relevance = self._compute_medical_relevance(complaint)
            
            # Combine scores (weighted average)
            combined_score = (
                0.4 * fluency_score +
                0.3 * coherence_score +
                0.3 * medical_relevance
            )
            
            scores.append(combined_score)
        
        return torch.tensor(scores, device=self.device, dtype=torch.float32)
    
    def _compute_fluency_score(self, text: str) -> float:
        """Compute fluency score based on language model perplexity."""
        if self.model is None:
            return 0.8 + 0.1 * np.random.randn()

        try:
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=512, padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

                # Use attention patterns as a proxy for fluency
                # More uniform attention often indicates more natural text
                attentions = outputs.attentions[-1] if hasattr(outputs, 'attentions') else None

                if attentions is not None:
                    # Compute attention entropy (higher entropy = more natural)
                    attention_probs = attentions.mean(dim=1)  # Average over heads
                    entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1)
                    fluency = torch.sigmoid(entropy.mean()).item()
                else:
                    # Fallback: use embedding magnitude as proxy
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    fluency = torch.sigmoid(embeddings.norm(dim=-1)).item()

            if self.perplexity_model is not None:
                with torch.no_grad():
                    ppl_inputs = self.perplexity_tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                        padding=True
                    ).to(self.device)
                    outputs = self.perplexity_model(
                        **ppl_inputs,
                        labels=ppl_inputs["input_ids"]
                    )
                    loss = outputs.loss.item()
                    perplexity = math.exp(min(loss, 20))
                    # Map perplexity (>=1) to [0,1], higher perplexity -> lower fluency
                    perplexity_score = 1.0 / (1.0 + (perplexity / 50.0))
                    fluency = 0.7 * fluency + 0.3 * perplexity_score

            return max(0.0, min(1.0, fluency))

        except Exception as e:
            self.logger.warning(f"Fluency computation failed: {e}")
            return 0.7
    
    def _compute_coherence_score(self, text: str) -> float:
        """Compute coherence score based on sentence-level consistency."""
        if self.model is None:
            return 0.8 + 0.1 * np.random.randn()
        
        try:
            # Split into sentences
            sentences = text.split('.')
            if len(sentences) < 2:
                return 0.8  # Single sentence, assume coherent
            
            # Encode each sentence
            sentence_embeddings = []
            for sentence in sentences[:3]:  # Limit to first 3 sentences
                sentence = sentence.strip()
                if len(sentence) < 5:
                    continue
                    
                inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True,
                                      max_length=128, padding=True).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    sentence_emb = outputs.last_hidden_state.mean(dim=1)
                    sentence_embeddings.append(sentence_emb)
            
            if len(sentence_embeddings) < 2:
                return 0.8
            
            # Compute pairwise similarities
            similarities = []
            for i in range(len(sentence_embeddings) - 1):
                sim = F.cosine_similarity(sentence_embeddings[i], sentence_embeddings[i+1])
                similarities.append(sim.item())
            
            # Higher similarity indicates better coherence
            coherence = np.mean(similarities)
            return max(0.0, min(1.0, coherence))
            
        except Exception as e:
            self.logger.warning(f"Coherence computation failed: {e}")
            return 0.7
    
    def _compute_medical_relevance(self, text: str) -> float:
        """Compute medical relevance score based on medical terminology presence."""
        # Medical keywords indicating authentic patient language
        medical_keywords = {
            'symptoms': ['pain', 'ache', 'hurt', 'sore', 'burning', 'stabbing', 'sharp', 'dull'],
            'breathing': ['breath', 'breathe', 'breathing', 'air', 'oxygen', 'suffocate', 'gasp'],
            'intensity': ['severe', 'mild', 'intense', 'terrible', 'awful', 'bad', 'worse'],
            'temporal': ['started', 'began', 'since', 'for', 'during', 'when', 'after', 'before'],
            'emotional': ['worried', 'scared', 'anxious', 'concerned', 'frightened', 'panicked'],
            'body_parts': ['chest', 'lung', 'throat', 'head', 'back', 'stomach', 'heart']
        }
        
        text_lower = text.lower()
        
        # Count medical keyword categories present
        category_scores = []
        for category, keywords in medical_keywords.items():
            present_keywords = sum(1 for keyword in keywords if keyword in text_lower)
            category_score = min(1.0, present_keywords / len(keywords))
            category_scores.append(category_score)
        
        # Compute weighted average
        weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.15]  # Symptoms and breathing weighted higher
        relevance = np.average(category_scores, weights=weights)
        
        # Boost score if text has personal pronouns (indicating patient perspective)
        personal_pronouns = ['i', 'my', 'me', 'myself']
        if any(pronoun in text_lower.split() for pronoun in personal_pronouns):
            relevance = min(1.0, relevance * 1.2)
        
        return relevance
    
    def get_detailed_scores(self, complaint: str) -> Dict[str, float]:
        """Get detailed breakdown of realism scores."""
        return {
            'fluency': self._compute_fluency_score(complaint),
            'coherence': self._compute_coherence_score(complaint),
            'medical_relevance': self._compute_medical_relevance(complaint),
            'overall': self.compute_realism_score([complaint]).item()
        }

class RealismLoss(nn.Module):
    """
    Loss function that incorporates realism scoring into generator training.
    """
    
    def __init__(self, realism_scorer: RealismScorer, weight: float = 1.0):
        super().__init__()
        self.realism_scorer = realism_scorer
        self.weight = weight
    
    def forward(self, complaints: List[str]) -> torch.Tensor:
        """
        Compute realism loss for generated complaints.
        
        Args:
            complaints: List of generated complaint texts
            
        Returns:
            Realism loss (lower when more realistic)
        """
        realism_scores = self.realism_scorer.compute_realism_score(complaints)
        
        # Convert to loss (higher realism = lower loss)
        # Use 1 - score so that higher realism gives lower loss
        realism_loss = (1.0 - realism_scores).mean()
        
        return self.weight * realism_loss

# Convenience function
def create_realism_scorer(model_name: str = "bert-base-uncased", 
                         use_medical_model: bool = True,
                         device: Optional[str] = None) -> RealismScorer:
    """Create a realism scorer instance."""
    return RealismScorer(model_name=model_name, 
                        use_medical_model=use_medical_model,
                        device=device)