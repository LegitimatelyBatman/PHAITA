"""Tests for the conversation engine dialogue controller."""

from phaita.conversation.engine import ConversationEngine


class SequenceGenerator:
    """Stub question generator that cycles through predefined prompts."""

    def __init__(self, questions):
        self.questions = questions
        self.index = 0

    def generate_clarifying_question(self, symptoms, **kwargs):
        question = self.questions[self.index % len(self.questions)]
        self.index += 1
        return question


class HistoryAwareGenerator:
    """Stub generator that ensures conversation history is provided."""

    def generate_clarifying_question(self, symptoms, previous_questions=None, **kwargs):
        previous_questions = previous_questions or []
        return f"Follow-up {len(previous_questions) + 1}"


def test_conversation_progression_tracks_symptoms():
    generator = SequenceGenerator([
        "When did the cough begin?",
        "Have you noticed a fever?",
        "Are there any other issues?",
    ])
    engine = ConversationEngine(generator, max_questions=5, min_symptom_count=3)
    engine.add_symptoms(["cough"])

    prompt1 = engine.next_prompt()
    assert prompt1 == "When did the cough begin?"
    engine.record_response(prompt1, "It started today and I now have a fever", ["fever"])
    assert not engine.should_present_diagnosis()

    prompt2 = engine.next_prompt()
    assert prompt2 == "Have you noticed a fever?"
    engine.record_response(prompt2, "I'm also feeling quite fatigued", ["fatigue"])

    assert engine.should_present_diagnosis()
    assert engine.symptoms == ["cough", "fever", "fatigue"]


def test_conversation_avoids_repeating_questions():
    generator = HistoryAwareGenerator()
    engine = ConversationEngine(generator, max_questions=4, min_symptom_count=5)
    engine.add_symptoms(["cough"])

    first = engine.next_prompt()
    engine.record_response(first, "No new symptoms", [])
    second = engine.next_prompt()

    assert first != second
    assert second == "Follow-up 2"


def test_conversation_termination_on_no_progress():
    generator = HistoryAwareGenerator()
    engine = ConversationEngine(generator, max_questions=5, min_symptom_count=6, max_no_progress_turns=2)
    engine.add_symptoms(["cough"])

    prompt1 = engine.next_prompt()
    engine.record_response(prompt1, "No additional issues", [])

    assert not engine.should_present_diagnosis()

    prompt2 = engine.next_prompt()
    engine.record_response(prompt2, "Still nothing new", [])

    assert engine.should_present_diagnosis()
    assert engine.next_prompt() is None
