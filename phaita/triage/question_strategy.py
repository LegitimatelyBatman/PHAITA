"""Question selection heuristics driven by expected information gain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..data.red_flags import RESPIRATORY_RED_FLAGS


def _normalise(text: str) -> str:
    return " ".join(text.lower().strip().split())


@dataclass
class ExpectedInformationGainStrategy:
    """Rank candidate prompts using a heuristic information gain metric."""

    red_flag_weight: float = 0.6
    symptom_weight: float = 0.3
    repeat_penalty: float = 0.5
    novelty_floor: float = 0.15

    def score_question(
        self,
        question: str,
        differential: Sequence[dict],
        conversation_history: Sequence[dict],
    ) -> float:
        """Compute expected information gain for a candidate question."""

        if not question:
            return 0.0

        question_norm = _normalise(question)
        asked = {_normalise(turn.get("question", "")) for turn in conversation_history}

        if question_norm in asked:
            return 0.0

        base_uncertainty = 0.0
        bonus = 0.0

        for entry in differential or []:
            probability = float(entry.get("probability", 0.0))
            if probability <= 0.0:
                continue

            # Encourage exploration where predictive uncertainty remains.
            base_uncertainty += probability * (1.0 - probability)

            code = entry.get("condition_code")
            evidence = entry.get("evidence", {})
            keywords: List[str] = []

            for symptom in evidence.get("key_symptoms", []) or []:
                keywords.append(symptom.replace("_", " "))

            for indicator in evidence.get("severity_indicators", []) or []:
                keywords.append(indicator.replace("_", " "))

            red_flag_bundle = RESPIRATORY_RED_FLAGS.get(code, {})
            for flag in red_flag_bundle.get("symptoms", []) or []:
                keywords.append(flag)

            for keyword in keywords:
                keyword_norm = _normalise(keyword)
                if keyword_norm and keyword_norm in question_norm:
                    weight = (
                        self.red_flag_weight
                        if keyword in red_flag_bundle.get("symptoms", [])
                        else self.symptom_weight
                    )
                    bonus += weight * probability

        # Scale base uncertainty by a novelty term so later turns naturally
        # diminish their marginal value without red-flag reinforcement.
        novelty_scale = 1.0 / (1.0 + max(len(conversation_history), 0))
        novelty_scale = max(novelty_scale, self.novelty_floor)

        info_gain = base_uncertainty * novelty_scale + bonus

        if question_norm in {_normalise(turn.get("question", "")) for turn in conversation_history[-2:]}:
            info_gain *= self.repeat_penalty

        return info_gain

    def select_question(
        self,
        candidates: Sequence[str],
        differential: Optional[Sequence[dict]],
        conversation_history: Sequence[dict],
    ) -> Tuple[Optional[str], float]:
        """Pick the question with the highest information gain score."""

        best_question: Optional[str] = None
        best_score = -1.0

        for candidate in candidates:
            score = self.score_question(candidate, differential or [], conversation_history)
            if score > best_score:
                best_question = candidate
                best_score = score

        if best_question is None:
            return None, 0.0

        return best_question, best_score if best_score > 0 else 0.0


__all__ = ["ExpectedInformationGainStrategy"]
