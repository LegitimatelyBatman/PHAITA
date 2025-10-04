#!/usr/bin/env python3
"""Interactive CLI for the patient simulator.

Install the project in editable mode (``pip install -e .``) before running the
CLI so that ``phaita`` imports resolve without manual path tweaks.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from phaita.data.icd_conditions import RespiratoryConditions
from phaita.generation.patient_agent import PatientPresentation, PatientSimulator
from phaita.models.generator import ComplaintGenerator


class ConversationIO:
    """Interface for reading clinician prompts and writing patient responses."""

    def display(self, text: str) -> None:
        raise NotImplementedError

    def prompt(self) -> Optional[str]:
        raise NotImplementedError


class ConsoleIO(ConversationIO):
    """Console-based implementation using :func:`input` and :func:`print`."""

    def __init__(self):
        self._prompt = "Clinician> "

    def display(self, text: str) -> None:  # pragma: no cover - trivial I/O wrapper
        print(text)

    def prompt(self) -> Optional[str]:  # pragma: no cover - trivial I/O wrapper
        try:
            return input(self._prompt)
        except EOFError:
            return None


@dataclass
class SessionResult:
    """Summary of a completed interview session."""

    presentation: PatientPresentation
    transcript: List[Tuple[str, str]]
    final_diagnosis: Optional[str]
    exit_reason: str


def _format_ground_truth_summary(
    presentation: PatientPresentation,
    *,
    exit_reason: str,
    final_diagnosis: Optional[str],
) -> List[str]:
    """Create a human-readable summary of the hidden presentation."""

    condition = RespiratoryConditions.get_condition_by_code(
        presentation.condition_code
    )
    lines = [""]
    lines.append("--- Ground Truth Reveal ---")
    if final_diagnosis:
        lines.append(f"Final Diagnosis Received: {final_diagnosis}")
    else:
        lines.append("Final Diagnosis Received: (none)")
    lines.append(f"Session Ended Via: {exit_reason}")
    lines.append(f"Condition Code: {presentation.condition_code}")
    lines.append(f"Condition Name: {condition['name']}")

    demographics = presentation.demographics
    lines.append("Demographics:")
    lines.append(f"  - Age: {demographics.age}")
    lines.append(f"  - Sex: {demographics.sex}")
    if demographics.ethnicity:
        lines.append(f"  - Ethnicity: {demographics.ethnicity}")
    if demographics.occupation:
        lines.append(f"  - Occupation: {demographics.occupation}")
    if demographics.social_history:
        lines.append("  - Social History: " + ", ".join(demographics.social_history))
    if demographics.risk_factors:
        lines.append("  - Risk Factors: " + ", ".join(demographics.risk_factors))
    if demographics.notes:
        lines.append("  - Additional Notes: " + ", ".join(demographics.notes))

    history = presentation.history_profile
    lines.append("History Highlights:")
    if history.past_conditions:
        lines.append("  - Past Conditions: " + ", ".join(history.past_conditions))
    if history.medications:
        lines.append("  - Medications: " + ", ".join(history.medications))
    if history.allergies:
        lines.append("  - Allergies: " + ", ".join(history.allergies))
    if history.last_meal:
        lines.append(f"  - Last Meal: {history.last_meal}")
    if history.recent_events:
        lines.append("  - Recent Events: " + ", ".join(history.recent_events))
    if history.family_history:
        lines.append("  - Family History: " + ", ".join(history.family_history))
    if history.lifestyle:
        lines.append("  - Lifestyle: " + ", ".join(history.lifestyle))
    if history.supports:
        lines.append("  - Supports: " + ", ".join(history.supports))
    if history.immunizations:
        lines.append("  - Immunizations: " + ", ".join(history.immunizations))

    true_symptoms = ", ".join(presentation.symptoms)
    lines.append(f"True Symptoms: {true_symptoms if true_symptoms else 'None'}")

    misdescribed = {
        symptom: overrides
        for symptom, overrides in presentation.vocabulary_profile.term_overrides.items()
        if overrides and any(value for value in overrides.values())
    }
    if misdescribed:
        lines.append("Misdescribed Symptoms:")
        for symptom, overrides in misdescribed.items():
            lines.append(f"  - {symptom}: {overrides}")
    else:
        lines.append("Misdescribed Symptoms: None")

    lines.append("Symptom Probabilities:")
    for symptom, probability in sorted(
        presentation.symptom_probabilities.items(), key=lambda item: item[0]
    ):
        lines.append(f"  - {symptom}: {probability:.2f}")

    lines.append("Misdescription Weights:")
    for symptom, weight in sorted(
        presentation.misdescription_weights.items(), key=lambda item: item[0]
    ):
        lines.append(f"  - {symptom}: {weight:.2f}")

    vocab = presentation.vocabulary_profile
    lines.append(f"Vocabulary Register: {vocab.register}")
    lines.append(f"Max Terms Per Response: {vocab.max_terms_per_response}")

    if presentation.follow_up_history:
        lines.append("Follow-up Exchanges:")
        for index, exchange in enumerate(presentation.follow_up_history, start=1):
            lines.append(f"  Q{index}: {exchange['prompt']}")
            lines.append(f"  A{index}: {exchange['response']}")
            mentions = exchange.get("symptom_mentions")
            if mentions is not None:
                mention_text = ", ".join(mentions) if mentions else "None"
                lines.append(f"     Mentioned Symptoms: {mention_text}")
    else:
        lines.append("No follow-up questions were asked.")

    return lines


def run_interview(
    io: ConversationIO,
    *,
    condition_code: Optional[str] = None,
    seed: Optional[int] = None,
    max_turns: Optional[int] = None,
    strategy: str = "default",
    max_terms: Optional[int] = None,
    simulator: Optional[PatientSimulator] = None,
    complaint_generator: Optional[ComplaintGenerator] = None,
) -> SessionResult:
    """Conduct an interview session and return the resulting summary."""

    if seed is not None:
        random.seed(seed)

    simulator = simulator or PatientSimulator()
    generator = complaint_generator or ComplaintGenerator(use_pretrained=False)

    if condition_code is None:
        condition_code, _ = RespiratoryConditions.get_random_condition()

    presentation = simulator.sample_presentation(condition_code)
    if max_terms is not None:
        presentation.vocabulary_profile.max_terms_per_response = max_terms
    presentation = generator.generate_complaint(presentation=presentation)

    transcript: List[Tuple[str, str]] = []
    opening = presentation.complaint_text or "I'm not feeling well."
    io.display(f"Patient: {opening}")
    transcript.append(("patient", opening))

    final_diagnosis: Optional[str] = None
    exit_reason = "completed"

    turns = 0
    while True:
        if max_turns is not None and turns >= max_turns:
            exit_reason = "max_turns"
            break

        prompt = io.prompt()
        if prompt is None:
            exit_reason = "eof"
            break

        prompt = prompt.strip()
        if not prompt:
            continue

        lowered = prompt.lower()
        if lowered in {"exit", "quit"}:
            exit_reason = "exit"
            break
        if lowered.startswith("diagnose"):
            diagnosis = prompt.split(" ", 1)
            final_diagnosis = diagnosis[1].strip() if len(diagnosis) > 1 else ""
            exit_reason = "diagnosis"
            break

        transcript.append(("clinician", prompt))
        response = generator.answer_question(prompt, strategy=strategy)
        transcript.append(("patient", response))
        io.display(f"Patient: {response}")
        turns += 1

    summary_lines = _format_ground_truth_summary(
        presentation,
        exit_reason=exit_reason,
        final_diagnosis=final_diagnosis if final_diagnosis else None,
    )
    for line in summary_lines:
        io.display(line)

    return SessionResult(
        presentation=presentation,
        transcript=transcript,
        final_diagnosis=final_diagnosis if final_diagnosis else None,
        exit_reason=exit_reason,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI entry point."""

    parser = argparse.ArgumentParser(description="Interact with a simulated patient")
    parser.add_argument(
        "--condition",
        help="ICD-10 condition code to simulate (defaults to random)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for deterministic simulation",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Maximum number of follow-up questions before auto-reveal",
    )
    parser.add_argument(
        "--strategy",
        choices=["default", "brief", "detailed"],
        default="default",
        help="Response strategy for follow-up questions",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        help="Override the vocabulary limit for symptom mentions per response",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> SessionResult:
    """Run the CLI using :class:`ConsoleIO`. Returns the session result."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    io = ConsoleIO()
    return run_interview(
        io,
        condition_code=args.condition,
        seed=args.seed,
        max_turns=args.max_turns,
        strategy=args.strategy,
        max_terms=args.max_terms,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
