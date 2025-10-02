"""Utilities for surfacing triage-ready differential diagnoses."""

from .diagnosis import (
    enrich_differential_with_guidance,
    format_differential_report,
)

__all__ = [
    "enrich_differential_with_guidance",
    "format_differential_report",
]
