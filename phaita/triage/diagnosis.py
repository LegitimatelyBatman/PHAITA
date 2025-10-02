"""Helpers for presenting diagnosis predictions with clinical guidance."""

from typing import Any, Dict, Iterable, List

from ..data.red_flags import RESPIRATORY_RED_FLAGS


def enrich_differential_with_guidance(
    ranked_predictions: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Attach red-flag and escalation guidance to ranked predictions.

    Args:
        ranked_predictions: Iterable of dictionaries produced by
            :meth:`phaita.models.discriminator.DiagnosisDiscriminator.predict_diagnosis`.

    Returns:
        A list of dictionaries with additional ``red_flags`` and
        ``escalation_advice`` keys.
    """

    enriched: List[Dict[str, Any]] = []
    for entry in ranked_predictions:
        code = entry.get("condition_code")
        red_flag_data = RESPIRATORY_RED_FLAGS.get(code, {})
        enriched_entry = dict(entry)
        enriched_entry["red_flags"] = red_flag_data.get("symptoms", [])
        enriched_entry["escalation_advice"] = red_flag_data.get("escalation", "")
        enriched.append(enriched_entry)

    return enriched


def format_differential_report(ranked_predictions: Iterable[Dict[str, Any]]) -> str:
    """Create a human-readable string summarising a ranked differential list."""

    enriched_predictions = enrich_differential_with_guidance(ranked_predictions)
    lines: List[str] = []

    for idx, prediction in enumerate(enriched_predictions, start=1):
        probability = prediction.get("probability", 0.0)
        ci_lower, ci_upper = prediction.get("confidence_interval", (0.0, 0.0))
        lines.append(
            f"{idx}. {prediction.get('condition_name')} "
            f"({prediction.get('condition_code')}) - "
            f"p={probability:.2f} (95% CI {ci_lower:.2f}-{ci_upper:.2f})"
        )

        evidence = prediction.get("evidence", {})
        key_symptoms = evidence.get("key_symptoms")
        severity_indicators = evidence.get("severity_indicators")
        description = evidence.get("description")

        if key_symptoms:
            lines.append(f"   🔎 Key symptoms: {', '.join(key_symptoms)}")
        if severity_indicators:
            lines.append(f"   🚨 Severe indicators: {', '.join(severity_indicators)}")
        if description:
            lines.append(f"   ℹ️  Summary: {description}")

        red_flags = prediction.get("red_flags")
        escalation = prediction.get("escalation_advice")
        if red_flags:
            lines.append(f"   ⚠️  Red flags: {', '.join(red_flags)}")
        if escalation:
            lines.append(f"   🚑 Escalation advice: {escalation}")

    return "\n".join(lines)
