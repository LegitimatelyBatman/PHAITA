"""Formatting helpers for clinician-ready patient info sheets."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ..data.icd_conditions import RespiratoryConditions


def _normalize_symptom(symptom: str) -> str:
    return symptom.strip().lower().replace(" ", "_")


def _extract_from_text(text: str) -> list[str]:
    segments = [segment for segment in text.split(",") if segment.strip()]
    return [
        symptom
        for segment in segments
        if (symptom := _normalize_symptom(segment))
    ]


def _format_symptom_list(symptoms: Iterable[str]) -> str:
    formatted = [symptom.replace("_", " ") for symptom in symptoms]
    return ", ".join(formatted) if formatted else "None"


def _detect_refuting_findings(
    conversation_turns: Sequence[Mapping[str, object]],
    candidate_symptoms: Sequence[str],
) -> list[str]:
    refuting: set[str] = set()
    lowered_candidates = {sym: sym.replace("_", " ") for sym in candidate_symptoms}

    for turn in conversation_turns:
        answer = str(turn.get("answer", "")).lower()
        for code, plain in lowered_candidates.items():
            if any(
                phrase in answer
                for phrase in (
                    f"no {plain}",
                    f"denies {plain}",
                    f"without {plain}",
                    f"not experiencing {plain}",
                )
            ):
                refuting.add(code)
    return sorted(refuting)


def format_info_sheet(
    conversation_turns: Sequence[Mapping[str, object]],
    ranked_predictions: Sequence[Mapping[str, object]],
    *,
    chief_complaint: str = "",
) -> str:
    """Generate a clinician-facing info sheet for the ranked differential."""

    observed_symptoms: set[str] = set()
    if chief_complaint:
        observed_symptoms.update(_extract_from_text(chief_complaint))
    for turn in conversation_turns:
        for symptom in turn.get("extracted_symptoms", []) or []:
            normalized = _normalize_symptom(str(symptom))
            if normalized:
                observed_symptoms.add(normalized)

    lines: list[str] = []
    if chief_complaint:
        lines.append(f"Chief complaint: {chief_complaint}")
        lines.append("")

    for idx, entry in enumerate(ranked_predictions, start=1):
        code = entry.get("condition_code", "UNKNOWN")
        try:
            condition = RespiratoryConditions.get_condition_by_code(code)
        except KeyError:
            condition = {"name": code, "symptoms": [], "severity_indicators": []}

        name = condition.get("name", code)
        condition_symptoms = [
            _normalize_symptom(symptom) for symptom in condition.get("symptoms", [])
        ]

        supporting = sorted(symptom for symptom in condition_symptoms if symptom in observed_symptoms)
        absent = sorted(symptom for symptom in condition_symptoms if symptom not in observed_symptoms)
        refuting = _detect_refuting_findings(conversation_turns, condition_symptoms)

        lines.append(f"{idx}. {name} ({code})")
        lines.append(f"   Supporting symptoms: {_format_symptom_list(supporting)}")
        lines.append(f"   Absent but expected: {_format_symptom_list(absent)}")
        lines.append(f"   Refuting findings: {_format_symptom_list(refuting)}")
        lines.append("   Clinician summary: ______________________________")
        lines.append("   Plan notes: _____________________________________")
        lines.append("")

    return "\n".join(lines).strip()
