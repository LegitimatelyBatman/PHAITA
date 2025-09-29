"""
PHAITA: Pre-Hospital AI Triage Algorithm
A medical triage system using adversarial training with Bayesian Networks,
Mistral 7B generator, and DeBERTa + GNN discriminator.
"""

__version__ = "0.1.0"
__author__ = "PHAITA Team"

# Only import data components (no dependencies)
from .data.icd_conditions import RespiratoryConditions

__all__ = [
    "RespiratoryConditions"
]