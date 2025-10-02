"""Patient simulation utilities wrapping the Bayesian symptom network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..models.bayesian_network import BayesianSymptomNetwork


@dataclass
class VocabularyProfile:
    """Simple representation of how a patient describes their symptoms."""

    allowed_terms: Optional[List[str]] = None
    term_overrides: Dict[str, Dict[str, str]] = field(default_factory=dict)
    register: str = "informal"
    max_terms_per_response: int = 3

    @classmethod
    def default_for(cls, symptoms: Sequence[str]) -> "VocabularyProfile":
        """Create a baseline vocabulary profile from canonical symptom names."""
        allowed = [symptom.replace("_", " ") for symptom in symptoms]
        overrides = {symptom: {} for symptom in symptoms}
        return cls(allowed_terms=allowed, term_overrides=overrides)

    def translate(self, symptom: str, *, form: str, default: str) -> str:
        """Translate a canonical symptom into the vocabulary for a given form."""
        override = self.term_overrides.get(symptom, {})
        candidate = override.get(form) if isinstance(override, dict) else None
        if not candidate:
            candidate = default
        candidate = candidate.replace("_", " ")
        if self.allowed_terms and candidate not in self.allowed_terms:
            # Restrict to allowed terms while keeping signal
            fallback = next((term for term in self.allowed_terms if term in candidate), None)
            if fallback:
                candidate = fallback
            else:
                candidate = self.allowed_terms[0]
        return candidate


@dataclass
class PatientPresentation:
    """Structured view of a simulated patient's presentation."""

    condition_code: str
    symptoms: List[str]
    symptom_probabilities: Dict[str, float]
    misdescription_weights: Dict[str, float]
    vocabulary_profile: VocabularyProfile
    complaint_text: Optional[str] = None
    follow_up_history: List[Dict[str, str]] = field(default_factory=list)

    def record_response(
        self,
        prompt: str,
        response: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an exchange to keep the simulation stateful."""

        entry: Dict[str, Any] = {"prompt": prompt, "response": response}
        if metadata:
            entry.update(metadata)
        self.follow_up_history.append(entry)


class PatientSimulator:
    """Thin wrapper around :class:`BayesianSymptomNetwork`."""

    def __init__(self, network: Optional[BayesianSymptomNetwork] = None):
        self.network = network or BayesianSymptomNetwork()

    def sample_presentation(
        self,
        condition_code: str,
        *,
        num_symptoms: Optional[int] = None,
        vocabulary_profile: Optional[VocabularyProfile] = None,
    ) -> PatientPresentation:
        """Sample a presentation and bundle metadata about symptom probabilities."""
        symptoms = self.network.sample_symptoms(condition_code, num_symptoms)
        probabilities = self.network.get_conditional_probabilities(condition_code)
        weights = {
            name: max(0.0, 1.0 - probabilities.get(name, 0.0))
            for name in probabilities.keys()
        }
        vocab = vocabulary_profile or VocabularyProfile.default_for(symptoms)
        return PatientPresentation(
            condition_code=condition_code,
            symptoms=symptoms,
            symptom_probabilities=probabilities,
            misdescription_weights=weights,
            vocabulary_profile=vocab,
        )

    def get_conditional_probabilities(
        self, condition_code: str, symptoms: Optional[Iterable[str]] = None
    ) -> Dict[str, float]:
        """Expose conditional probabilities for external consumers."""
        probabilities = self.network.get_conditional_probabilities(condition_code)
        if symptoms is None:
            return probabilities
        return {symptom: probabilities.get(symptom, 0.0) for symptom in symptoms}
