"""Dialogue controller for managing clarifying conversations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set

from ..triage.question_strategy import ExpectedInformationGainStrategy
from ..utils.text import normalize_symptom_to_underscores


@dataclass
class ConversationTurn:
    """Single question/response exchange within a conversation."""

    question: str
    response: Optional[str] = None
    extracted_symptoms: List[str] = field(default_factory=list)


class ConversationEngine:
    """State machine that manages a clarifying dialogue.

    The engine keeps track of gathered symptoms, pending prompts that still
    require answers, and heuristics that determine when enough information has
    been collected to surface diagnoses.
    """

    def __init__(
        self,
        question_generator,
        *,
        max_questions: int = 6,
        min_symptom_count: int = 3,
        max_no_progress_turns: int = 2,
        max_generation_attempts: int = 3,
        candidate_pool_size: int = 3,
        information_gain_threshold: float = 0.02,
        question_selection_strategy: Optional[ExpectedInformationGainStrategy] = None,
    ) -> None:
        self.question_generator = question_generator
        self.max_questions = max_questions
        self.min_symptom_count = min_symptom_count
        self.max_no_progress_turns = max_no_progress_turns
        self.max_generation_attempts = max_generation_attempts
        self.candidate_pool_size = candidate_pool_size
        self.information_gain_threshold = information_gain_threshold
        self.question_strategy = question_selection_strategy or ExpectedInformationGainStrategy()

        self._symptom_order: List[str] = []
        self._symptom_set: Set[str] = set()
        self.turns: List[ConversationTurn] = []
        self.unanswered_prompts: Deque[str] = deque()
        self.no_progress_turns: int = 0
        self._stopped: bool = False
        self._info_gain_history: List[float] = []
        self._current_differential: Sequence[dict] = []
        self.last_info_gain_gradient: Optional[float] = None
        self.demographics_context: Optional[Dict[str, Any]] = None
        self.history_context: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Symptom tracking helpers
    # ------------------------------------------------------------------
    def add_symptoms(self, symptoms: Iterable[str]) -> None:
        """Merge newly observed symptoms into the conversation state."""

        for symptom in symptoms:
            if not symptom:
                continue
            # Normalize symptom name to underscores format
            normalized = normalize_symptom_to_underscores(symptom)
            if not normalized:
                continue
            if normalized not in self._symptom_set:
                self._symptom_set.add(normalized)
                self._symptom_order.append(normalized)

    @property
    def symptoms(self) -> List[str]:
        """Return gathered symptoms in the order they were discovered."""

        return list(self._symptom_order)

    def set_patient_context(
        self,
        *,
        demographics: Optional[Dict[str, Any]] = None,
        history: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store demographic and history context for downstream question generation."""

        self.demographics_context = demographics
        self.history_context = history

    # ------------------------------------------------------------------
    # Conversation flow
    # ------------------------------------------------------------------
    def next_prompt(self) -> Optional[str]:
        """Return the next question to ask or ``None`` when conversation ends."""

        if self.should_present_diagnosis():
            return None

        if self.unanswered_prompts:
            return self.unanswered_prompts[0]

        if len(self.turns) >= self.max_questions:
            self._stopped = True
            return None

        attempts = 0
        asked_questions = {turn.question for turn in self.turns}
        history = [
            {"question": turn.question, "answer": turn.response or ""}
            for turn in self.turns
        ]
        previous_answers = [turn.response or "" for turn in self.turns]
        previous_questions = [turn.question for turn in self.turns]

        while attempts < self.max_generation_attempts:
            if hasattr(self.question_generator, "generate_candidate_questions"):
                candidates = self.question_generator.generate_candidate_questions(
                    self.symptoms,
                    previous_answers=previous_answers,
                    previous_questions=previous_questions,
                    conversation_history=history,
                    demographics=self.demographics_context,
                    history=self.history_context,
                    num_candidates=self.candidate_pool_size,
                )
            else:
                candidate = self.question_generator.generate_clarifying_question(
                    self.symptoms,
                    previous_answers=previous_answers,
                    previous_questions=previous_questions,
                    conversation_history=history,
                    demographics=self.demographics_context,
                    history=self.history_context,
                )
                candidates = [candidate] if candidate else []

            selected_question, info_gain = self.question_strategy.select_question(
                candidates,
                self._current_differential,
                history,
            )
            attempts += 1

            if selected_question:
                normalized_question = selected_question.strip()
                if normalized_question and normalized_question not in asked_questions:
                    info_gain_value = info_gain or 0.0
                    if self._info_gain_history:
                        previous_gain = self._info_gain_history[-1]
                        gradient = previous_gain - info_gain_value
                        self.last_info_gain_gradient = gradient
                        if (
                            previous_gain > 0
                            and info_gain_value > 0
                            and gradient >= 0
                            and gradient <= self.information_gain_threshold
                        ):
                            self._stopped = True
                            return None
                    else:
                        self.last_info_gain_gradient = None

                    self._info_gain_history.append(info_gain_value)
                    turn = ConversationTurn(question=normalized_question)
                    self.turns.append(turn)
                    self.unanswered_prompts.append(normalized_question)
                    return normalized_question

            # Avoid infinite loops by updating attempts and history
            previous_questions = [turn.question for turn in self.turns]

        self._stopped = True
        return None

    def record_response(
        self,
        question: str,
        response: str,
        extracted_symptoms: Optional[Sequence[str]] = None,
    ) -> None:
        """Record the patient's response to a prompt."""

        for turn in self.turns:
            if turn.question == question:
                turn.response = response
                break
        else:
            # If the question wasn't known, treat as historical and append.
            turn = ConversationTurn(question=question, response=response)
            self.turns.append(turn)

        if self.unanswered_prompts and self.unanswered_prompts[0] == question:
            self.unanswered_prompts.popleft()
        else:
            # Remove matching question if it exists elsewhere in the queue
            try:
                self.unanswered_prompts.remove(question)
            except ValueError:
                pass

        extracted = list(extracted_symptoms or [])
        turn.extracted_symptoms = extracted

        previous_symptom_count = len(self._symptom_set)
        self.add_symptoms(extracted)

        if len(self._symptom_set) > previous_symptom_count:
            self.no_progress_turns = 0
        else:
            self.no_progress_turns += 1

        if self.no_progress_turns >= self.max_no_progress_turns:
            self._stopped = True

    # ------------------------------------------------------------------
    # Termination logic
    # ------------------------------------------------------------------
    def should_present_diagnosis(self) -> bool:
        """Check if enough information has been gathered."""

        if self._stopped:
            return True

        if len(self.symptoms) >= self.min_symptom_count:
            return True

        if len(self.turns) >= self.max_questions and not self.unanswered_prompts:
            return True

        return False

    def has_pending_prompt(self) -> bool:
        """Return True when a question still awaits an answer."""

        return bool(self.unanswered_prompts)

    def reset(self) -> None:
        """Reset the engine to its initial state."""

        self._symptom_order.clear()
        self._symptom_set.clear()
        self.turns.clear()
        self.unanswered_prompts.clear()
        self.no_progress_turns = 0
        self._stopped = False
        self._info_gain_history.clear()
        self._current_differential = []
        self.last_info_gain_gradient = None

    # ------------------------------------------------------------------
    # Differential management
    # ------------------------------------------------------------------
    def update_differential(self, ranked_predictions: Sequence[dict]) -> None:
        """Update the latest differential used for question selection."""

        self._current_differential = list(ranked_predictions or [])

    @property
    def information_gain_history(self) -> List[float]:
        """Return a copy of the historical information gain values."""

        return list(self._info_gain_history)
