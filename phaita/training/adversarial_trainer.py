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
from ..utils.metrics import compute_diversity_metrics, compute_diagnosis_metrics


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
    """
    
    def __init__(self, 
                 generator_lr: float = 2e-5,
                 discriminator_lr: float = 1e-4,
                 diversity_weight: float = 0.1,
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
        self.condition_codes = list(RespiratoryConditions.get_all_conditions().keys())
        self.training_history = defaultdict(list)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
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
        Train generator for one step (adversarial + diversity).
        
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
        
        # Total generator loss
        gen_loss = gen_adv_loss + self.diversity_weight * diversity_loss
        
        gen_loss.backward()
        self.gen_optimizer.step()
        
        return {
            "gen_loss": gen_loss.item(),
            "gen_adv_loss": gen_adv_loss.item(),
            "diversity_loss": diversity_loss.item()
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
            
            # Training steps
            for step in range(10):  # Multiple steps per epoch
                
                # Generate training data
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
            
            # Logging
            if epoch % eval_interval == 0:
                # Generate evaluation data
                eval_complaints, eval_codes, _ = self.generate_training_batch(32)
                eval_metrics = self.evaluate(eval_complaints, eval_codes)
                
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