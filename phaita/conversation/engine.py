"""Dialogue controller for managing clarifying conversations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Sequence, Set


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
    ) -> None:
        self.question_generator = question_generator
        self.max_questions = max_questions
        self.min_symptom_count = min_symptom_count
        self.max_no_progress_turns = max_no_progress_turns
        self.max_generation_attempts = max_generation_attempts

        self._symptom_order: List[str] = []
        self._symptom_set: Set[str] = set()
        self.turns: List[ConversationTurn] = []
        self.unanswered_prompts: Deque[str] = deque()
        self.no_progress_turns: int = 0
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # Symptom tracking helpers
    # ------------------------------------------------------------------
    def add_symptoms(self, symptoms: Iterable[str]) -> None:
        """Merge newly observed symptoms into the conversation state."""

        for symptom in symptoms:
            if not symptom:
                continue
            normalized = symptom.strip().lower().replace(" ", "_")
            if not normalized:
                continue
            if normalized not in self._symptom_set:
                self._symptom_set.add(normalized)
                self._symptom_order.append(normalized)

    @property
    def symptoms(self) -> List[str]:
        """Return gathered symptoms in the order they were discovered."""

        return list(self._symptom_order)

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
            question = self.question_generator.generate_clarifying_question(
                self.symptoms,
                previous_answers=previous_answers,
                previous_questions=previous_questions,
                conversation_history=history,
            )
            attempts += 1

            if question:
                normalized_question = question.strip()
                if normalized_question and normalized_question not in asked_questions:
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
