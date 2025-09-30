"""
PHAITA: Pre-Hospital AI Triage Algorithm
A medical triage system using adversarial training with Bayesian Networks,
Mistral 7B generator, and DeBERTa + GNN discriminator.
"""

__version__ = "0.1.0"
__author__ = "PHAITA Team"

# Import data components
from .data.icd_conditions import RespiratoryConditions

# Import model components
from .models.generator import SymptomGenerator, ComplaintGenerator
from .models.discriminator import DiagnosisDiscriminator
from .models.bayesian_network import BayesianSymptomNetwork

# Import training components
try:
    from .training.adversarial_trainer import AdversarialTrainer
except ImportError:
    # Fallback for when torch is not available
    class AdversarialTrainer:
        def __init__(self, *args, **kwargs):
            pass

# Import config
try:
    from .utils.config import Config
except ImportError:
    # Fallback for when dependencies are not available
    class Config:
        def __init__(self):
            pass
        
        @classmethod
        def from_yaml(cls, path):
            return cls()

__all__ = [
    "RespiratoryConditions",
    "SymptomGenerator",
    "ComplaintGenerator",
    "DiagnosisDiscriminator",
    "BayesianSymptomNetwork",
    "AdversarialTrainer", 
    "Config"
]