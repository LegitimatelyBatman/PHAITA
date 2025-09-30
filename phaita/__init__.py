"""
PHAITA: Pre-Hospital AI Triage Algorithm
A medical triage system using adversarial training with Bayesian Networks,
Mistral 7B generator, and DeBERTa + GNN discriminator.
"""

__version__ = "0.1.0"
__author__ = "PHAITA Team"

# Only import data components (no dependencies)
from .data.icd_conditions import RespiratoryConditions

# Mock classes for CLI compatibility
class AdversarialTrainer:
    def __init__(self, *args, **kwargs):
        pass

class Config:
    def __init__(self):
        pass
    
    @classmethod
    def from_yaml(cls, path):
        return cls()

class SymptomGenerator:
    def __init__(self):
        pass

class ComplaintGenerator:  
    def __init__(self):
        pass

class DiagnosisDiscriminator:
    def __init__(self):
        pass

__all__ = [
    "RespiratoryConditions",
    "AdversarialTrainer", 
    "Config",
    "SymptomGenerator",
    "ComplaintGenerator", 
    "DiagnosisDiscriminator"
]