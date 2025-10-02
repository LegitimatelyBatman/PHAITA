"""
PHAITA Data Layer - Medical conditions and data processing utilities.
"""

from .icd_conditions import RespiratoryConditions
from .red_flags import RESPIRATORY_RED_FLAGS

__all__ = ["RespiratoryConditions", "RESPIRATORY_RED_FLAGS"]
