"""
Adversarial training implementation for medical triage system.
Implements diversity loss to prevent repetition and improve generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm
import logging
from collections import defaultdict

from ..models.generator import SymptomGenerator, ComplaintGenerator
from ..models.discriminator import DiagnosisDiscriminator
from ..data.icd_conditions import RespiratoryConditions
from ..data.forum_scraper import create_data_augmentation, ForumDataAugmentation
from ..utils.metrics import compute_diversity_metrics, compute_diagnosis_metrics
from ..utils.realism_scorer import create_realism_scorer, RealismLoss


class DiversityLoss(nn.Module):
    """
    Diversity loss to prevent repetition in generated complaints.
    Encourages semantic and lexical diversity.
    """
    
    def __init__(self, lambda_semantic: float = 1.0, lambda_lexical: float = 0.5):
        super().__init__()
        self.lambda_semantic = lambda_semantic
        self.lambda_lexical = lambda_lexical
    
    def compute_semantic_diversity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic diversity loss based on embedding similarities.
        
        Args:
            embeddings: [batch_size, embed_dim] tensor of text embeddings
        
        Returns:
            Diversity loss (lower when more diverse)
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise cosine similarities
        similarities = torch.mm(embeddings, embeddings.t())
        
        # Remove diagonal (self-similarities)
        mask = ~torch.eye(similarities.size(0), dtype=torch.bool, device=similarities.device)
        similarities = similarities[mask]
        
        # Diversity loss is the mean similarity (want to minimize)
        diversity_loss = similarities.mean()
        
        return diversity_loss
    
    def compute_lexical_diversity(self, texts: List[str]) -> float:
        """
        Compute lexical diversity based on unique word ratios.
        
        Args:
            texts: List of generated complaint texts
        
        Returns:
            Lexical diversity score
        """
        if not texts:
            return 0.0
        
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        # Compute type-token ratio (unique words / total words)
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        diversity_score = unique_words / total_words
        
        # Convert to loss (want to maximize diversity, so minimize 1 - diversity)
        diversity_loss = 1.0 - diversity_score
        
        return diversity_loss
    
    def forward(self, embeddings: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        Compute total diversity loss.
        
        Args:
            embeddings: Text embeddings from DeBERTa
            texts: Generated complaint texts
        
        Returns:
            Combined diversity loss
        """
        semantic_loss = self.compute_semantic_diversity(embeddings)
        lexical_loss = self.compute_lexical_diversity(texts)
        
        total_loss = (self.lambda_semantic * semantic_loss + 
                     self.lambda_lexical * lexical_loss)
        
        return total_loss


class AdversarialTrainer:
    """
    Main adversarial training loop for the medical triage system.
    Includes curriculum learning and forum data integration.
    """
    
    def __init__(self, 
                 generator_lr: float = 2e-5,
                 discriminator_lr: float = 1e-4,
                 diversity_weight: float = 0.1,
                 realism_weight: float = 0.1,
                 use_curriculum_learning: bool = True,
                 use_forum_data: bool = True,
                 device: Optional[str] = None):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.symptom_generator = SymptomGenerator()
        self.complaint_generator = ComplaintGenerator()
        self.discriminator = DiagnosisDiscriminator().to(self.device)
        
        # Initialize loss functions
        self.diversity_loss = DiversityLoss()
        self.diagnosis_loss = nn.CrossEntropyLoss()
        self.adversarial_loss = nn.BCELoss()
        
        # Initialize realism scorer and loss
        self.realism_scorer = create_realism_scorer(use_medical_model=True, device=self.device)
        self.realism_loss = RealismLoss(self.realism_scorer, weight=realism_weight)
        
        # Initialize optimizers
        self.gen_optimizer = AdamW(
            self.discriminator.parameters(),  # Only text encoder is trainable
            lr=generator_lr,
            weight_decay=0.01
        )
        
        self.disc_optimizer = AdamW(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            weight_decay=0.01
        )
        
        # Learning rate schedulers
        self.gen_scheduler = None
        self.disc_scheduler = None
        
        # Training state
        self.diversity_weight = diversity_weight
        self.realism_weight = realism_weight
        self.condition_codes = list(RespiratoryConditions.get_all_conditions().keys())
        self.training_history = defaultdict(list)
        
        # Curriculum learning parameters
        self.use_curriculum_learning = use_curriculum_learning
        self.curriculum_stage = 0  # 0: synthetic only, 1: mixed, 2: forum heavy
        self.curriculum_schedule = [0.0, 0.3, 0.7]  # Forum data ratio by stage
        
        # Forum data integration
        self.use_forum_data = use_forum_data
        self.forum_augmenter = create_data_augmentation() if use_forum_data else None
        self.forum_complaints = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load forum data if enabled
        if self.use_forum_data:
            self._load_forum_data()
    
    def _load_forum_data(self):
        """Load forum complaints for curriculum learning."""
        try:
            self.forum_complaints = self.forum_augmenter.get_forum_complaints_for_pretraining(max_complaints=1000)
            self.logger.info(f"Loaded {len(self.forum_complaints)} forum complaints for curriculum learning")
        except Exception as e:
            self.logger.warning(f"Failed to load forum data: {e}")
            self.forum_complaints = []
    
    def _get_curriculum_forum_ratio(self, epoch: int, total_epochs: int) -> float:
        """Get the ratio of forum data to use based on curriculum learning schedule."""
        if not self.use_curriculum_learning or not self.forum_complaints:
            return 0.0
        
        # Progress through curriculum stages
        progress = epoch / total_epochs
        
        if progress < 0.3:  # Stage 0: Synthetic only
            return 0.0
        elif progress < 0.7:  # Stage 1: Mixed
            return 0.3
        else:  # Stage 2: Forum heavy
            return 0.7
    
    def _sample_mixed_training_data(self, batch_size: int, forum_ratio: float) -> Tuple[List[str], torch.Tensor]:
        """Sample mixed synthetic and forum data based on curriculum."""
        forum_count = int(batch_size * forum_ratio)
        synthetic_count = batch_size - forum_count
        
        all_complaints = []
        all_labels = []
        
        # Generate synthetic data
        if synthetic_count > 0:
            synthetic_complaints, condition_codes, synthetic_labels = self.generate_training_batch(synthetic_count)
            all_complaints.extend(synthetic_complaints)
            all_labels.extend(synthetic_labels.tolist())
        
        # Sample forum data
        if forum_count > 0 and self.forum_complaints:
            forum_subset = random.sample(self.forum_complaints, min(forum_count, len(self.forum_complaints)))
            all_complaints.extend(forum_subset)
            
            # For forum data, assign random labels (as they're unlabeled)
            forum_labels = [random.randint(0, len(self.condition_codes) - 1) for _ in range(len(forum_subset))]
            all_labels.extend(forum_labels)
        
        # Convert labels to tensor
        labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=self.device)
        
        return all_complaints, labels_tensor
    
    def generate_training_batch(self, batch_size: int) -> Tuple[List[str], List[str], torch.Tensor]:
        """
        Generate a batch of synthetic training data.
        
        Returns:
            complaints: Generated patient complaints
            condition_codes: Corresponding condition codes
            labels: One-hot encoded labels for conditions
        """
        complaints = []
        condition_codes = []
        
        for _ in range(batch_size):
            # Sample random condition
            condition_code = random.choice(self.condition_codes)
            condition_codes.append(condition_code)
            
            # Generate symptoms
            symptoms = self.symptom_generator.bayesian_network.sample_symptoms(condition_code)
            
            # Generate patient complaint
            complaint = self.complaint_generator.generate_complaint(symptoms, condition_code)
            complaints.append(complaint)
        
        # Create labels
        labels = []
        for code in condition_codes:
            label_idx = self.condition_codes.index(code)
            labels.append(label_idx)
        
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        return complaints, condition_codes, labels
    
    def train_discriminator_step(self, real_complaints: List[str], real_labels: torch.Tensor,
                                fake_complaints: List[str]) -> Dict[str, float]:
        """
        Train discriminator for one step.
        
        Args:
            real_complaints: Real patient complaints (or high-quality synthetic)
            real_labels: True condition labels
            fake_complaints: Generated complaints from current generator
        
        Returns:
            Dictionary of loss values
        """
        self.disc_optimizer.zero_grad()
        
        # Process real data
        real_outputs = self.discriminator(real_complaints)
        real_diagnosis_loss = self.diagnosis_loss(real_outputs["diagnosis_logits"], real_labels)
        
        # Real samples should be classified as real (label = 1)
        real_labels_adv = torch.ones(len(real_complaints), 1, device=self.device)
        real_adv_loss = self.adversarial_loss(real_outputs["discriminator_scores"], real_labels_adv)
        
        # Process fake data
        fake_outputs = self.discriminator(fake_complaints)
        
        # Fake samples should be classified as fake (label = 0)
        fake_labels_adv = torch.zeros(len(fake_complaints), 1, device=self.device)
        fake_adv_loss = self.adversarial_loss(fake_outputs["discriminator_scores"], fake_labels_adv)
        
        # Total discriminator loss
        disc_loss = real_diagnosis_loss + real_adv_loss + fake_adv_loss
        
        disc_loss.backward()
        self.disc_optimizer.step()
        
        return {
            "disc_loss": disc_loss.item(),
            "real_diagnosis_loss": real_diagnosis_loss.item(),
            "real_adv_loss": real_adv_loss.item(),
            "fake_adv_loss": fake_adv_loss.item()
        }
    
    def train_generator_step(self, batch_size: int) -> Dict[str, float]:
        """
        Train generator for one step (adversarial + diversity + realism).
        
        Args:
            batch_size: Size of generated batch
        
        Returns:
            Dictionary of loss values
        """
        self.gen_optimizer.zero_grad()
        
        # Generate fake complaints
        fake_complaints, _, _ = self.generate_training_batch(batch_size)
        
        # Get discriminator outputs
        fake_outputs = self.discriminator(fake_complaints, return_features=True)
        
        # Generator wants fake samples to be classified as real (label = 1)
        real_labels_adv = torch.ones(len(fake_complaints), 1, device=self.device)
        gen_adv_loss = self.adversarial_loss(fake_outputs["discriminator_scores"], real_labels_adv)
        
        # Diversity loss to encourage varied complaints
        diversity_loss = self.diversity_loss(fake_outputs["text_features"], fake_complaints)
        
        # Realism loss to encourage authentic-sounding complaints
        realism_loss = self.realism_loss(fake_complaints)
        
        # Total generator loss
        gen_loss = (gen_adv_loss + 
                   self.diversity_weight * diversity_loss + 
                   self.realism_weight * realism_loss)
        
        gen_loss.backward()
        self.gen_optimizer.step()
        
        return {
            "gen_loss": gen_loss.item(),
            "gen_adv_loss": gen_adv_loss.item(),
            "diversity_loss": diversity_loss.item(),
            "realism_loss": realism_loss.item()
        }
    
    def evaluate(self, eval_complaints: List[str], eval_labels: List[str]) -> Dict[str, float]:
        """
        Evaluate the current model performance.
        
        Args:
            eval_complaints: Evaluation complaint texts
            eval_labels: True condition codes
        
        Returns:
            Evaluation metrics
        """
        self.discriminator.eval()
        
        with torch.no_grad():
            predictions = self.discriminator.predict_diagnosis(eval_complaints)
            predicted_codes = [pred[0] for pred in predictions]
            confidences = [pred[1] for pred in predictions]
        
        # Compute metrics
        metrics = compute_diagnosis_metrics(eval_labels, predicted_codes, confidences)
        
        # Compute diversity metrics for generated complaints
        generated_complaints, _, _ = self.generate_training_batch(len(eval_complaints))
        diversity_metrics = compute_diversity_metrics(generated_complaints)
        
        metrics.update(diversity_metrics)
        
        self.discriminator.train()
        return metrics
    
    def train(self, 
              num_epochs: int = 100,
              batch_size: int = 16,
              eval_interval: int = 10,
              save_interval: int = 50) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            eval_interval: Evaluate every N epochs
            save_interval: Save model every N epochs
        
        Returns:
            Training history
        """
        self.logger.info(f"Starting adversarial training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        
        # Initialize schedulers
        self.gen_scheduler = CosineAnnealingLR(self.gen_optimizer, T_max=num_epochs)
        self.disc_scheduler = CosineAnnealingLR(self.disc_optimizer, T_max=num_epochs)
        
        for epoch in range(num_epochs):
            epoch_losses = defaultdict(float)
            
            # Get curriculum learning forum ratio
            forum_ratio = self._get_curriculum_forum_ratio(epoch, num_epochs)
            
            # Training steps
            for step in range(10):  # Multiple steps per epoch
                
                # Generate mixed training data based on curriculum
                if self.use_curriculum_learning and self.forum_complaints:
                    real_complaints, real_labels = self._sample_mixed_training_data(batch_size, forum_ratio)
                    fake_complaints, _, fake_labels = self.generate_training_batch(batch_size)
                else:
                    # Generate training data (original approach)
                    fake_complaints, _, fake_labels = self.generate_training_batch(batch_size)
                    real_complaints = fake_complaints  # Using synthetic as "real" for now
                    real_labels = fake_labels
                
                # Train discriminator
                disc_losses = self.train_discriminator_step(
                    real_complaints, real_labels, fake_complaints
                )
                
                # Train generator (less frequently to balance training)
                if step % 2 == 0:
                    gen_losses = self.train_generator_step(batch_size)
                    for key, value in gen_losses.items():
                        epoch_losses[key] += value
                
                for key, value in disc_losses.items():
                    epoch_losses[key] += value
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= 10
                self.training_history[key].append(epoch_losses[key])
            
            # Update learning rates
            self.gen_scheduler.step()
            self.disc_scheduler.step()
            
            # Enhanced logging and evaluation
            if epoch % eval_interval == 0:
                # Generate evaluation data
                eval_complaints, eval_codes, _ = self.generate_training_batch(32)
                eval_metrics = self.evaluate(eval_complaints, eval_codes)
                
                # Compute additional metrics
                diversity_metrics = self._compute_epoch_diversity_metrics(eval_complaints)
                realism_metrics = self._compute_epoch_realism_metrics(eval_complaints)
                medical_consistency = self._compute_medical_consistency(eval_complaints, eval_codes)
                
                # Log comprehensive metrics
                self.logger.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Gen Loss: {epoch_losses.get('gen_loss', 0):.4f} | "
                    f"Disc Loss: {epoch_losses.get('disc_loss', 0):.4f} | "
                    f"Diversity: {diversity_metrics['avg_diversity']:.4f} | "
                    f"Realism: {realism_metrics['avg_realism']:.4f} | "
                    f"Medical Consistency: {medical_consistency:.4f} | "
                    f"Forum Ratio: {forum_ratio:.2f}"
                )
                
                # Store evaluation metrics
                for key, value in eval_metrics.items():
                    self.training_history[f"eval_{key}"].append(value)
                
                # Store additional metrics
                self.training_history["diversity_score"].append(diversity_metrics['avg_diversity'])
                self.training_history["realism_score"].append(realism_metrics['avg_realism'])
                self.training_history["medical_consistency"].append(medical_consistency)
                self.training_history["forum_ratio"].append(forum_ratio)
                
                # Log adversarial failure cases
                self._log_adversarial_failures(eval_complaints, eval_metrics)
                
                self.logger.info(f"Epoch {epoch}")
                self.logger.info(f"  Disc Loss: {epoch_losses['disc_loss']:.4f}")
                self.logger.info(f"  Gen Loss: {epoch_losses.get('gen_loss', 0):.4f}")
                self.logger.info(f"  Diversity Loss: {epoch_losses.get('diversity_loss', 0):.4f}")
                self.logger.info(f"  Eval Accuracy: {eval_metrics.get('accuracy', 0):.4f}")
                self.logger.info(f"  Lexical Diversity: {eval_metrics.get('lexical_diversity', 0):.4f}")
            
            # Save model
            if epoch % save_interval == 0 and epoch > 0:
                self.save_models(f"epoch_{epoch}")
        
        self.logger.info("Training completed!")
        return dict(self.training_history)
    
    def _compute_epoch_diversity_metrics(self, complaints: List[str]) -> Dict[str, float]:
        """Compute diversity metrics for current epoch."""
        if not complaints:
            return {'avg_diversity': 0.0}
        
        # Lexical diversity (unique words / total words)
        all_words = []
        for complaint in complaints:
            words = complaint.lower().split()
            all_words.extend(words)
        
        if not all_words:
            return {'avg_diversity': 0.0}
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        lexical_diversity = unique_words / total_words if total_words > 0 else 0.0
        
        # Semantic diversity (average pairwise distance)
        if len(complaints) < 2:
            semantic_diversity = 1.0
        else:
            # Simple semantic diversity based on complaint length variety
            lengths = [len(complaint.split()) for complaint in complaints]
            mean_length = np.mean(lengths)
            length_variance = np.var(lengths) if len(lengths) > 1 else 0.0
            semantic_diversity = min(1.0, length_variance / (mean_length + 1e-8))
        
        avg_diversity = (lexical_diversity + semantic_diversity) / 2
        
        return {
            'avg_diversity': avg_diversity,
            'lexical_diversity': lexical_diversity,
            'semantic_diversity': semantic_diversity
        }
    
    def _compute_epoch_realism_metrics(self, complaints: List[str]) -> Dict[str, float]:
        """Compute realism metrics for current epoch."""
        if not complaints:
            return {'avg_realism': 0.0}
        
        # Use realism scorer to get detailed scores
        total_realism = 0.0
        fluency_scores = []
        coherence_scores = []
        medical_relevance_scores = []
        
        for complaint in complaints[:10]:  # Sample to avoid slowdown
            scores = self.realism_scorer.get_detailed_scores(complaint)
            fluency_scores.append(scores['fluency'])
            coherence_scores.append(scores['coherence'])
            medical_relevance_scores.append(scores['medical_relevance'])
            total_realism += scores['overall']
        
        avg_realism = total_realism / len(complaints[:10])
        
        return {
            'avg_realism': avg_realism,
            'avg_fluency': np.mean(fluency_scores),
            'avg_coherence': np.mean(coherence_scores),
            'avg_medical_relevance': np.mean(medical_relevance_scores)
        }
    
    def _compute_medical_consistency(self, complaints: List[str], condition_codes: List[str]) -> float:
        """Compute medical consistency between complaints and assigned conditions."""
        if not complaints or not condition_codes:
            return 0.0
        
        consistency_scores = []
        
        for complaint, code in zip(complaints, condition_codes):
            # Get expected symptoms for condition
            try:
                condition_data = RespiratoryConditions.get_condition_by_code(code)
                expected_symptoms = condition_data['symptoms'] + condition_data['severity_indicators']
                lay_terms = condition_data['lay_terms']
                
                # Check if complaint contains relevant terms
                complaint_lower = complaint.lower()
                
                symptom_matches = sum(1 for symptom in expected_symptoms 
                                    if any(word in complaint_lower for word in symptom.split()))
                lay_matches = sum(1 for term in lay_terms 
                                if term.lower() in complaint_lower)
                
                total_matches = symptom_matches + lay_matches
                total_possible = len(expected_symptoms) + len(lay_terms)
                
                consistency = total_matches / total_possible if total_possible > 0 else 0.0
                consistency_scores.append(consistency)
                
            except Exception as e:
                self.logger.warning(f"Error computing medical consistency: {e}")
                consistency_scores.append(0.0)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _log_adversarial_failures(self, complaints: List[str], eval_metrics: Dict[str, float]):
        """Log cases where generator successfully fools discriminator."""
        # This would require discriminator predictions, simplifying for now
        failure_rate = 1.0 - eval_metrics.get('accuracy', 0.0)
        
        if failure_rate > 0.8:  # High failure rate
            self.logger.warning(f"High adversarial failure rate: {failure_rate:.3f}")
            
            # Log sample failures
            sample_complaints = complaints[:3]
            for i, complaint in enumerate(sample_complaints):
                self.logger.warning(f"Sample failure {i+1}: {complaint[:100]}...")
    
    def pretrain_discriminator_on_forum_data(self, num_epochs: int = 10, batch_size: int = 16):
        """Pretrain discriminator on forum data before adversarial training."""
        if not self.use_forum_data or not self.forum_complaints:
            self.logger.info("Skipping forum data pretraining - no forum data available")
            return
        
        self.logger.info(f"Pretraining discriminator on forum data for {num_epochs} epochs")
        
        # Create simple optimizer for pretraining
        pretrain_optimizer = AdamW(self.discriminator.parameters(), lr=1e-4)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Create batches from forum complaints
            for i in range(0, len(self.forum_complaints), batch_size):
                batch_complaints = self.forum_complaints[i:i+batch_size]
                
                if len(batch_complaints) < 2:
                    continue
                
                # Assign random labels (unsupervised pretraining)
                batch_labels = torch.randint(0, len(self.condition_codes), 
                                           (len(batch_complaints),), device=self.device)
                
                pretrain_optimizer.zero_grad()
                
                # Forward pass
                outputs = self.discriminator(batch_complaints)
                loss = self.diagnosis_loss(outputs["diagnosis_logits"], batch_labels)
                
                # Backward pass
                loss.backward()
                pretrain_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            if epoch % 5 == 0:
                self.logger.info(f"Pretrain Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        self.logger.info("Forum data pretraining completed")
    
    def save_models(self, checkpoint_name: str):
        """Save model checkpoints."""
        import os
        os.makedirs("checkpoints", exist_ok=True)
        
        torch.save({
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'training_history': dict(self.training_history),
        }, f"checkpoints/{checkpoint_name}.pth")
        
        self.logger.info(f"Models saved as {checkpoint_name}.pth")
    
    def load_models(self, checkpoint_path: str):
        """Load model checkpoints."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        self.logger.info(f"Models loaded from {checkpoint_path}")