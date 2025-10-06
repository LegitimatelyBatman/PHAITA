"""
Learnable comorbidity effects module.

Makes comorbidity symptom modifiers learnable through PyTorch parameters
instead of fixed weights from YAML configuration.
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LearnableComorbidityEffects(nn.Module):
    """
    Learnable comorbidity effects with PyTorch parameters.
    
    Initializes weights from YAML config but makes them learnable through
    gradient descent during training. Each comorbidity has a set of symptom
    modifiers that are parameterized as learnable weights.
    """
    
    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize learnable comorbidity effects.
        
        Args:
            config_path: Path to comorbidity_effects.yaml (optional)
            device: PyTorch device ('cpu' or 'cuda')
        """
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load initial values from YAML
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "comorbidity_effects.yaml"
        
        self.config_data = self._load_config(config_path)
        
        # Build vocabulary of all symptoms that can be affected by comorbidities
        self.symptom_vocab = {}
        self.comorbidity_vocab = {}
        self._build_vocabularies()
        
        # Create learnable parameter matrix: [num_comorbidities, num_symptoms]
        # Each entry is a log-multiplier (so exp(param) gives the multiplier)
        num_comorbidities = len(self.comorbidity_vocab)
        num_symptoms = len(self.symptom_vocab)
        
        # Initialize from config values
        initial_weights = torch.zeros(num_comorbidities, num_symptoms, device=self.device)
        for comorbidity, comorbidity_idx in self.comorbidity_vocab.items():
            if comorbidity in self.config_data:
                modifiers = self.config_data[comorbidity].get('symptom_modifiers', {})
                for symptom, multiplier in modifiers.items():
                    if symptom in self.symptom_vocab:
                        symptom_idx = self.symptom_vocab[symptom]
                        # Store as log(multiplier) so we can exponentiate later
                        # This ensures multipliers stay positive
                        initial_weights[comorbidity_idx, symptom_idx] = torch.log(torch.tensor(multiplier))
        
        # Make it a learnable parameter
        self.comorbidity_weights = nn.Parameter(initial_weights)
        
        # Store specific symptoms and probabilities (these remain fixed for now)
        self.specific_symptoms = {}
        self.specific_symptom_probs = {}
        for comorbidity in self.comorbidity_vocab:
            if comorbidity in self.config_data:
                self.specific_symptoms[comorbidity] = self.config_data[comorbidity].get('specific_symptoms', [])
                self.specific_symptom_probs[comorbidity] = self.config_data[comorbidity].get('probability', 0.3)
        
        # Interaction effects (keep as dictionary for now, could be learned later)
        self.interaction_effects = self.config_data.get('interactions', {})
        self.max_symptom_probability = self.config_data.get('max_probability', 0.95)
    
    def _load_config(self, config_path: Path) -> Dict:
        """Load comorbidity configuration from YAML."""
        default_config = {
            "diabetes": {
                "symptom_modifiers": {"fatigue": 1.3, "infection_risk": 1.5},
                "specific_symptoms": [],
                "probability": 0.3
            },
            "hypertension": {
                "symptom_modifiers": {"dyspnea": 1.2, "chest_pain": 1.4},
                "specific_symptoms": [],
                "probability": 0.3
            },
            "obesity": {
                "symptom_modifiers": {"shortness_of_breath": 1.5, "exercise_intolerance": 1.3},
                "specific_symptoms": [],
                "probability": 0.3
            },
            "max_probability": 0.95,
            "interactions": {}
        }
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    # Filter out non-dict entries and interactions
                    filtered_config = {}
                    for key, value in config_data.items():
                        if isinstance(value, dict) and 'symptom_modifiers' in value:
                            filtered_config[key] = value
                    # Add special keys
                    if 'interactions' in config_data:
                        filtered_config['interactions'] = config_data['interactions']
                    if 'max_probability' in config_data:
                        filtered_config['max_probability'] = config_data['max_probability']
                    return filtered_config
        except Exception:
            pass
        
        return default_config
    
    def _build_vocabularies(self):
        """Build vocabulary mappings for comorbidities and symptoms."""
        all_symptoms = set()
        comorbidity_list = []
        
        for comorbidity, data in self.config_data.items():
            if isinstance(data, dict) and 'symptom_modifiers' in data:
                comorbidity_list.append(comorbidity)
                modifiers = data.get('symptom_modifiers', {})
                all_symptoms.update(modifiers.keys())
        
        # Create index mappings
        self.comorbidity_vocab = {c: i for i, c in enumerate(sorted(comorbidity_list))}
        self.symptom_vocab = {s: i for i, s in enumerate(sorted(all_symptoms))}
    
    def get_symptom_modifiers(self, comorbidity: str) -> Dict[str, float]:
        """
        Get symptom modifiers for a comorbidity using learned weights.
        
        Args:
            comorbidity: Name of the comorbidity
            
        Returns:
            Dictionary mapping symptoms to their learned multipliers
        """
        if comorbidity not in self.comorbidity_vocab:
            return {}
        
        comorbidity_idx = self.comorbidity_vocab[comorbidity]
        modifiers = {}
        
        for symptom, symptom_idx in self.symptom_vocab.items():
            # Exponentiate to get positive multiplier
            multiplier = torch.exp(self.comorbidity_weights[comorbidity_idx, symptom_idx]).item()
            # Only include non-trivial multipliers (not exactly 1.0)
            if abs(multiplier - 1.0) > 0.01:
                modifiers[symptom] = multiplier
        
        return modifiers
    
    def get_comorbidity_data(self, comorbidity: str) -> Dict:
        """
        Get all comorbidity data including learned modifiers and fixed metadata.
        
        Args:
            comorbidity: Name of the comorbidity
            
        Returns:
            Dictionary with 'symptom_modifiers', 'specific_symptoms', 'probability'
        """
        return {
            'symptom_modifiers': self.get_symptom_modifiers(comorbidity),
            'specific_symptoms': self.specific_symptoms.get(comorbidity, []),
            'probability': self.specific_symptom_probs.get(comorbidity, 0.3)
        }
    
    def get_all_comorbidities(self) -> List[str]:
        """Get list of all known comorbidities."""
        return list(self.comorbidity_vocab.keys())
    
    def forward(self) -> torch.Tensor:
        """
        Forward pass returns the weight matrix for gradient computation.
        
        Returns:
            Comorbidity weight matrix with exp applied for positivity
        """
        return torch.exp(self.comorbidity_weights)


def create_learnable_comorbidity_effects(config_path: Optional[str] = None, device: Optional[str] = None) -> LearnableComorbidityEffects:
    """
    Factory function to create a learnable comorbidity effects module.
    
    Args:
        config_path: Optional path to comorbidity_effects.yaml
        device: Optional device ('cpu' or 'cuda')
        
    Returns:
        Initialized LearnableComorbidityEffects module
    """
    return LearnableComorbidityEffects(config_path=config_path, device=device)
