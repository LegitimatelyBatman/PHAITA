"""
PHAITA Models - Bayesian networks, generators, and discriminators.
"""

from .bayesian_network import BayesianSymptomNetwork
from .generator import SymptomGenerator, ComplaintGenerator
from .discriminator import DiagnosisDiscriminator
from .discriminator_lite import LightweightDiscriminator

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

# Import learnable modules (optional - requires torch)
try:
    from .learnable_comorbidity import LearnableComorbidityEffects, create_learnable_comorbidity_effects
    from .learnable_causality import LearnableSymptomCausality, create_learnable_causality
    _LEARNABLE_AVAILABLE = True
except ImportError:
    _LEARNABLE_AVAILABLE = False
    LearnableComorbidityEffects = None
    LearnableSymptomCausality = None
    create_learnable_comorbidity_effects = None
    create_learnable_causality = None

__all__ = [
    "BayesianSymptomNetwork",
    "SymptomGenerator",
    "ComplaintGenerator",
    "DiagnosisDiscriminator",
    "LightweightDiscriminator",
    "SymptomTimeline",
    "TemporalSymptomEncoder",
    "TemporalPatternMatcher",
    "LearnableComorbidityEffects",
    "LearnableSymptomCausality",
    "create_learnable_comorbidity_effects",
    "create_learnable_causality",
]
