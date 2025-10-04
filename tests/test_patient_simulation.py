import random

import pytest

from phaita.generation.patient_agent import (
    PatientSimulator,
    VocabularyProfile,
)
from phaita.models.generator import ComplaintGenerator, SymptomGenerator


def test_patient_presentation_metadata_consistency():
    simulator = PatientSimulator()
    presentation = simulator.sample_presentation("J45.9")

    assert presentation.condition_code == "J45.9"
    assert presentation.symptoms, "Expected sampled symptoms"
    assert presentation.symptom_probabilities
    assert presentation.misdescription_weights
    assert presentation.demographics.age > 0
    assert presentation.demographics.sex
    assert presentation.history_profile is not None
    assert presentation.demographic_criteria
    assert presentation.history_criteria

    for symptom, probability in presentation.symptom_probabilities.items():
        assert 0.0 <= probability <= 1.0
        weight = presentation.misdescription_weights.get(symptom)
        assert weight is not None
        assert weight == pytest.approx(max(0.0, 1.0 - probability), rel=1e-5)


def test_complaint_generator_follow_up_respects_vocabulary():
    random.seed(0)
    symptom_generator = SymptomGenerator()
    complaint_generator = ComplaintGenerator(use_pretrained=False)

    vocabulary = VocabularyProfile.default_for(["cough", "shortness_of_breath"])
    presentation = symptom_generator.generate_symptoms(
        "J45.9", vocabulary_profile=vocabulary
    )
    presentation = complaint_generator.generate_complaint(presentation=presentation)

    assert presentation.complaint_text

    response = complaint_generator.answer_question(
        "How long have you felt this way?", strategy="detailed"
    )

    # Follow-up responses should be recorded
    assert presentation.follow_up_history
    assert presentation.follow_up_history[-1]["response"] == response

    # Ensure vocabulary constraints are respected
    allowed_terms = set(vocabulary.allowed_terms or [])
    if allowed_terms:
        assert any(term in response for term in allowed_terms)
    for symptom in presentation.symptoms:
        if "_" in symptom:
            assert symptom not in response

    medication_response = complaint_generator.answer_question(
        "What medications are you taking right now?"
    )
    assert "Rescue inhaler" in medication_response or "not on any" in medication_response
