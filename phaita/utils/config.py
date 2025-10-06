"""
Configuration management for PHAITA system.

Configuration Files:
- config/system.yaml: Global technical configuration (model, training, data)
- config/medical_knowledge.yaml: Physician-editable medical knowledge
- config/templates.yaml: Complaint generation templates

For backward compatibility, also supports legacy config.yaml and individual config files.
"""

import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
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
class ConversationConfig:
    """Configuration for conversation/dialogue parameters."""
    max_questions: int = 10
    confidence_threshold: float = 0.85
    min_info_gain: float = 0.1
    enable_red_flag_escalation: bool = True


@dataclass
class TriageConfig:
    """Configuration for triage/diagnosis parameters."""
    max_diagnoses: int = 10
    min_confidence: float = 0.05
    enable_red_flag_check: bool = True
    enable_info_sheets: bool = True
    escalation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'critical': 0.95,
        'urgent': 0.80,
        'routine': 0.50
    })


@dataclass
class Config:
    """Main configuration class.
    
    Loads from config/system.yaml by default, with backward compatibility
    for legacy config.yaml.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    triage: TriageConfig = field(default_factory=TriageConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to config file. If None, tries:
                1. config/system.yaml (new structure)
                2. config.yaml (legacy)
        
        Returns:
            Loaded Config object
        """
        # Determine which config file to use
        if yaml_path is None:
            # Try new structure first
            project_root = Path(__file__).resolve().parents[2]
            system_config = project_root / "config" / "system.yaml"
            legacy_config = project_root / "config.yaml"
            
            if system_config.exists():
                yaml_path = str(system_config)
            elif legacy_config.exists():
                yaml_path = str(legacy_config)
            else:
                # No config file, use defaults
                return cls()
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'conversation' in config_dict:
            config.conversation = ConversationConfig(**config_dict['conversation'])
        if 'triage' in config_dict:
            triage_dict = config_dict['triage']
            config.triage = TriageConfig(**triage_dict)
        
        return config
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'conversation': asdict(self.conversation),
            'triage': asdict(self.triage)
        }
        
        os.makedirs(os.path.dirname(yaml_path) or '.', exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (ModelConfig, TrainingConfig, DataConfig, 
                                                   ConversationConfig, TriageConfig)):
                    # Update nested config
                    for nested_key, nested_value in value.items():
                        setattr(getattr(self, key), nested_key, nested_value)
                else:
                    setattr(self, key, value)


def get_medical_config_path() -> Path:
    """Get the path to the medical knowledge configuration file.
    
    Returns:
        Path to config/medical_knowledge.yaml
    """
    return Path(__file__).resolve().parents[2] / "config" / "medical_knowledge.yaml"


def load_medical_config() -> Dict[str, Any]:
    """Load medical knowledge configuration.
    
    This includes conditions, red_flags, comorbidity_effects,
    symptom_causality, and temporal_patterns.
    
    Returns:
        Dictionary with medical configuration data
    """
    config_path = get_medical_config_path()
    
    if not config_path.exists():
        # Fall back to loading individual files for backward compatibility
        return _load_legacy_medical_configs()
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _load_legacy_medical_configs() -> Dict[str, Any]:
    """Load medical configs from legacy individual files.
    
    For backward compatibility with old config structure.
    """
    config_dir = Path(__file__).resolve().parents[2] / "config"
    result = {}
    
    # Try to load each component
    for component in ['conditions', 'red_flags', 'comorbidity_effects', 
                      'symptom_causality', 'temporal_patterns']:
        legacy_name = component.replace('_', '_')  # respiratory_conditions -> respiratory_conditions
        if component == 'conditions':
            legacy_name = 'respiratory_conditions'
        
        legacy_path = config_dir / f"{legacy_name}.yaml"
        if legacy_path.exists():
            with open(legacy_path, 'r') as f:
                result[component] = yaml.safe_load(f)
    
    return result