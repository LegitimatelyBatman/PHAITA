"""Tests for triage differential formatting and guidance."""

from phaita.models.discriminator import DiagnosisDiscriminator
from phaita.triage import enrich_differential_with_guidance, format_differential_report


def test_predict_diagnosis_honours_top_k_and_guidance():
    model = DiagnosisDiscriminator(use_pretrained=False)

    complaint = "I have a cough, fever, and my chest hurts when I breathe."
    top_k = 4

    differential_lists = model.predict_diagnosis([complaint], top_k=top_k)
    assert len(differential_lists) == 1

    ranked = differential_lists[0]
    assert len(ranked) == top_k

    # Probabilities should be in non-increasing order
    probs = [entry["probability"] for entry in ranked]
    assert probs == sorted(probs, reverse=True)

    # Confidence interval should be a pair of floats inside [0, 1]
    lower, upper = ranked[0]["confidence_interval"]
    assert 0.0 <= lower <= upper <= 1.0

    # Enriched guidance should include red flags and escalation messaging
    enriched = enrich_differential_with_guidance(ranked)
    assert enriched[0]["red_flags"], "Primary entry missing red-flag symptoms"
    assert enriched[0]["escalation_advice"], "Primary entry missing escalation advice"

    report = format_differential_report(ranked)
    assert "Red flags" in report


def test_predict_with_explanation_returns_matched_keywords():
    model = DiagnosisDiscriminator(use_pretrained=False)

    complaint = "I have a cough, fever, and my chest hurts when I breathe."

    code, confidence, explanation = model.predict_with_explanation(complaint)

    assert code in model.condition_codes
    assert 0.0 <= confidence <= 1.0

    matched_keywords = explanation.get("matched_keywords")
    assert isinstance(matched_keywords, list)
    assert matched_keywords, "Expected at least one matched keyword in explanation"
    for keyword in matched_keywords:
        assert keyword in complaint.lower()
