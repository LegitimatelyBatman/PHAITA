"""Regression tests for the conversation engine's triage flow.

Note: This test suite uses stub generators and does NOT require real transformer models.
It validates the conversation engine logic independently of model implementation.
For tests requiring real models, see test_integration.py.
"""

from phaita.conversation.engine import ConversationEngine


class SequenceGenerator:
    """Stub generator that yields a deterministic sequence of prompts."""

    def __init__(self, questions):
        self.questions = list(questions)
        self.index = 0

    def generate_candidate_questions(self, symptoms, **kwargs):  # noqa: D401
        question = self.questions[self.index % len(self.questions)]
        self.index += 1
        return [question]

    def generate_clarifying_question(self, symptoms, **kwargs):
        return self.generate_candidate_questions(symptoms, **kwargs)[0]


class HistoryAwareGenerator:
    """Stub generator that inspects prior questions for variety."""

    def __init__(self):
        self.counter = 0

    def generate_candidate_questions(self, symptoms, previous_questions=None, **kwargs):
        self.counter += 1
        label = previous_questions or []
        return [f"Follow-up {len(label) + 1}"]

    def generate_clarifying_question(self, symptoms, **kwargs):
        return self.generate_candidate_questions(symptoms, **kwargs)[0]


class CandidatePoolGenerator:
    """Generator that exposes a precomputed pool of candidate prompts."""

    def __init__(self, pools):
        self.pools = list(pools)
        self.index = 0

    def generate_candidate_questions(self, symptoms, **kwargs):
        pool = self.pools[self.index % len(self.pools)]
        self.index += 1
        return list(pool)

    def generate_clarifying_question(self, symptoms, **kwargs):
        return self.generate_candidate_questions(symptoms, **kwargs)[0]


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
    engine = ConversationEngine(
        generator,
        max_questions=5,
        min_symptom_count=6,
        max_no_progress_turns=2,
    )
    engine.add_symptoms(["cough"])

    prompt1 = engine.next_prompt()
    engine.record_response(prompt1, "No additional issues", [])

    assert not engine.should_present_diagnosis()

    prompt2 = engine.next_prompt()
    engine.record_response(prompt2, "Still nothing new", [])

    assert engine.should_present_diagnosis()
    assert engine.next_prompt() is None


def test_engine_prioritises_red_flag_questions():
    generator = CandidatePoolGenerator([
        [
            "Have you noticed bluish lips or fingernails recently?",
            "When did your coughing episodes begin?",
        ]
    ])
    engine = ConversationEngine(
        generator,
        max_questions=3,
        min_symptom_count=5,
        information_gain_threshold=0.01,
    )
    engine.add_symptoms(["cough"])

    differential = [
        {
            "condition_code": "J45.9",
            "probability": 0.7,
            "evidence": {
                "key_symptoms": ["wheezing"],
                "severity_indicators": ["breathlessness at rest"],
            },
        }
    ]
    engine.update_differential(differential)

    prompt = engine.next_prompt()

    assert "bluish lips" in prompt.lower()
    assert engine.information_gain_history and engine.information_gain_history[0] > 0


def test_engine_information_gain_threshold_stops_loop():
    generator = SequenceGenerator([
        "Can you describe your breathing?",
        "Do you have chest tightness?",
        "Any other concerns?",
    ])
    engine = ConversationEngine(
        generator,
        max_questions=6,
        min_symptom_count=5,
        information_gain_threshold=0.05,
    )
    engine.add_symptoms(["cough"])

    differential = [
        {
            "condition_code": "J06.9",
            "probability": 0.5,
            "evidence": {
                "key_symptoms": ["cough"],
                "severity_indicators": ["fever"],
            },
        }
    ]

    engine.update_differential(differential)

    first = engine.next_prompt()
    engine.record_response(first, "Nothing new", [])
    engine.update_differential(differential)

    second = engine.next_prompt()
    engine.record_response(second, "Still nothing", [])
    engine.update_differential(differential)

    # The third call should stop due to diminishing information gain
    third = engine.next_prompt()

    assert third is None
    assert engine.should_present_diagnosis()
    assert engine.information_gain_history[-1] > 0
    assert engine.last_info_gain_gradient is not None
