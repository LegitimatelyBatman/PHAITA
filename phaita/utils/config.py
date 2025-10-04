"""
Configuration management for PHAITA system.
"""

import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
import os


@dataclass
class ModelConfig:
    """Configuration for model architectures."""
    deberta_model: str = "microsoft/deberta-base"
    mistral_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    use_quantization: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_epochs: int = 100
    batch_size: int = 16
    generator_lr: float = 2e-5
    discriminator_lr: float = 1e-4
    diversity_weight: float = 0.1
    eval_interval: int = 10
    save_interval: int = 50
    device: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for data parameters."""
    num_respiratory_conditions: int = 10
    min_symptoms_per_condition: int = 3
    max_symptoms_per_condition: int = 7
    num_complaint_variants: int = 3


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        
        return config
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data)
        }
        
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (ModelConfig, TrainingConfig, DataConfig)):
                    # Update nested config
                    for nested_key, nested_value in value.items():
                        setattr(getattr(self, key), nested_key, nested_value)
                else:
                    setattr(self, key, value)