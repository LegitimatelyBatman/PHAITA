"""Patient simulation utilities wrapping the Bayesian symptom network."""

from __future__ import annotations

import random
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
class PatientDemographics:
    """Demographic snapshot for a simulated patient."""

    age: int = 40
    sex: str = "unspecified"
    ethnicity: Optional[str] = None
    occupation: Optional[str] = None
    social_history: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.age}-year-old"]
        if self.sex and self.sex != "unspecified":
            parts.append(self.sex)
        if self.ethnicity and self.ethnicity.lower() != "any":
            parts.append(self.ethnicity)
        if self.occupation:
            parts.append(self.occupation)
        return " ".join(parts).strip()


@dataclass
class PatientHistory:
    """Medical history snapshot for a simulated patient."""

    past_conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    last_meal: Optional[str] = None
    recent_events: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)
    lifestyle: List[str] = field(default_factory=list)
    immunizations: List[str] = field(default_factory=list)
    supports: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "past_conditions": list(self.past_conditions),
            "medications": list(self.medications),
            "allergies": list(self.allergies),
            "last_meal": self.last_meal,
            "recent_events": list(self.recent_events),
            "family_history": list(self.family_history),
            "lifestyle": list(self.lifestyle),
            "immunizations": list(self.immunizations),
            "supports": list(self.supports),
        }


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
    demographics: PatientDemographics = field(default_factory=PatientDemographics)
    history_profile: PatientHistory = field(default_factory=PatientHistory)
    demographic_criteria: Dict[str, Any] = field(default_factory=dict)
    history_criteria: Dict[str, Any] = field(default_factory=dict)

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
        condition_data = self.network.conditions.get(condition_code, {})
        demographics, demographic_criteria = self._sample_demographics(condition_data)
        history_profile, history_criteria = self._sample_history(condition_data)
        vocab = vocabulary_profile or VocabularyProfile.default_for(symptoms)
        return PatientPresentation(
            condition_code=condition_code,
            symptoms=symptoms,
            symptom_probabilities=probabilities,
            misdescription_weights=weights,
            vocabulary_profile=vocab,
            demographics=demographics,
            history_profile=history_profile,
            demographic_criteria=demographic_criteria,
            history_criteria=history_criteria,
        )

    def get_conditional_probabilities(
        self, condition_code: str, symptoms: Optional[Iterable[str]] = None
    ) -> Dict[str, float]:
        """Expose conditional probabilities for external consumers."""
        probabilities = self.network.get_conditional_probabilities(condition_code)
        if symptoms is None:
            return probabilities
        return {symptom: probabilities.get(symptom, 0.0) for symptom in symptoms}

    def _sample_demographics(
        self, condition_data: Dict[str, Any]
    ) -> tuple[PatientDemographics, Dict[str, Any]]:
        criteria = condition_data.get("demographics", {}) or {"inclusion": {}, "exclusion": {}}
        inclusion = criteria.get("inclusion", {})

        age_ranges = inclusion.get("age_ranges") or [
            {"min": 18.0, "max": 75.0, "weight": 1.0}
        ]
        weights = [max(entry.get("weight", 1.0), 1e-3) for entry in age_ranges]
        selected_range = random.choices(age_ranges, weights=weights, k=1)[0]
        minimum = int(selected_range.get("min", 0))
        maximum = int(selected_range.get("max", minimum))
        if maximum < minimum:
            maximum = minimum
        age = random.randint(minimum, maximum)

        sexes = inclusion.get("sexes") or ["unspecified"]
        sex = random.choice(sexes)

        ethnicity = None
        if inclusion.get("ethnicities"):
            ethnicity = random.choice(inclusion["ethnicities"])

        occupation = None
        if inclusion.get("occupations"):
            occupation = random.choice(inclusion["occupations"])

        social_history = list(inclusion.get("social_history", []))
        risk_factors = list(inclusion.get("risk_factors", [])) + list(
            inclusion.get("exposures", [])
        )
        notes = list(inclusion.get("notes", [])) + list(inclusion.get("regions", []))

        demographics = PatientDemographics(
            age=age,
            sex=sex,
            ethnicity=ethnicity,
            occupation=occupation,
            social_history=social_history,
            risk_factors=risk_factors,
            notes=notes,
        )

        return demographics, criteria

    def _sample_history(
        self, condition_data: Dict[str, Any]
    ) -> tuple[PatientHistory, Dict[str, Any]]:
        criteria = condition_data.get("history", {}) or {"inclusion": {}, "exclusion": {}}
        inclusion = criteria.get("inclusion", {})

        history = PatientHistory(
            past_conditions=list(inclusion.get("past_conditions", [])),
            medications=list(inclusion.get("medications", [])),
            allergies=list(inclusion.get("allergies", [])),
            last_meal=random.choice(inclusion.get("last_meal", [None])) if inclusion.get("last_meal") else None,
            recent_events=list(inclusion.get("recent_events", [])),
            family_history=list(inclusion.get("family_history", [])),
            lifestyle=list(inclusion.get("lifestyle", [])),
            immunizations=list(inclusion.get("immunizations", [])),
            supports=list(inclusion.get("supports", [])),
        )

        return history, criteria
