"""
PHAITA Models - Bayesian networks, generators, and discriminators.
"""

from .bayesian_network import BayesianSymptomNetwork
from .generator import SymptomGenerator, ComplaintGenerator
from .discriminator import DiagnosisDiscriminator

# Import temporal module components (optional - requires torch)
try:
    from .temporal_module import (
        SymptomTimeline,
        TemporalSymptomEncoder,
        TemporalPatternMatcher,
    )
    _TEMPORAL_AVAILABLE = True
except ImportError:
    _TEMPORAL_AVAILABLE = False
    SymptomTimeline = None
    TemporalSymptomEncoder = None
    TemporalPatternMatcher = None

__all__ = [
    "BayesianSymptomNetwork",
    "SymptomGenerator",
    "ComplaintGenerator",
    "DiagnosisDiscriminator",
    "SymptomTimeline",
    "TemporalSymptomEncoder",
    "TemporalPatternMatcher",
]
