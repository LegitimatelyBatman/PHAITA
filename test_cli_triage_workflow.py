from types import SimpleNamespace as Namespace
from unittest.mock import MagicMock, patch

import cli


class StubQuestionGenerator:
    def __init__(self, *args, **kwargs):
        self._calls = 0

    def generate_clarifying_question(
        self,
        symptoms,
        previous_answers=None,
        previous_questions=None,
        conversation_history=None,
        **kwargs,
    ):
        if self._calls == 0:
            self._calls += 1
            return "Are you feeling shortness of breath?"
        self._calls += 1
        return None


def test_diagnose_command_runs_information_gain_loop(capsys):
    args = Namespace(complaint="wheezing, cough", detailed=False, interactive=False)

    discriminator = MagicMock()
    discriminator.predict_diagnosis.side_effect = [
        [[
            {
                "condition_code": "J45.9",
                "condition_name": "Asthma",
                "probability": 0.60,
                "confidence_interval": (0.40, 0.80),
            }
        ]],
        [[
            {
                "condition_code": "J45.9",
                "condition_name": "Asthma",
                "probability": 0.75,
                "confidence_interval": (0.55, 0.90),
            },
            {
                "condition_code": "J18.9",
                "condition_name": "Pneumonia",
                "probability": 0.12,
                "confidence_interval": (0.05, 0.25),
            },
        ]],
    ]

    input_responses = iter(["shortness of breath"])

    cli._get_diagnosis_discriminator.cache_clear()

    with (
        patch.object(cli, "_get_diagnosis_discriminator", return_value=discriminator),
        patch.object(cli, "QuestionGenerator", StubQuestionGenerator),
        patch.object(cli, "input", side_effect=lambda prompt: next(input_responses, "quit")),
    ):
        cli.diagnose_command(args)

    output = capsys.readouterr().out

    assert "p=0.60" in output
    assert discriminator.predict_diagnosis.call_count >= 2
    assert "Asthma (J45.9)" in output
    assert "Supporting symptoms:" in output
    assert "Absent but expected:" in output
    assert "Refuting findings:" in output


class StubChallengeQuestionGenerator(StubQuestionGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._responses = [
            "Does the patient have fever?",
            "Any chest pain?",
            None,
        ]

    def generate_clarifying_question(self, *args, **kwargs):
        if self._responses:
            return self._responses.pop(0)
        return None


def test_challenge_command_uses_information_loop(capsys):
    args = Namespace(rare_cases=1, atypical_cases=0, show_failures=False, verbose=True)

    class FakeNetwork:
        def sample_symptoms(self, condition_code, **kwargs):
            metadata = {"presentation_type": "rare", "case_name": f"Rare {condition_code}"}
            metadata.update(kwargs)
            return ["cough", "fever", "chest_pain"], metadata

    class FakeComplaintGenerator:
        def generate_complaint(self, symptoms, condition_code):
            return f"complaint for {condition_code}"

    discriminator = MagicMock()
    discriminator.predict_diagnosis.side_effect = [
        [[
            {
                "condition_code": "J18.9",
                "condition_name": "Pneumonia",
                "probability": 0.55,
                "confidence_interval": (0.30, 0.70),
            },
            {
                "condition_code": "J45.9",
                "condition_name": "Asthma",
                "probability": 0.18,
                "confidence_interval": (0.05, 0.40),
            },
        ]],
        [[
            {
                "condition_code": "J18.9",
                "condition_name": "Pneumonia",
                "probability": 0.62,
                "confidence_interval": (0.35, 0.78),
            },
            {
                "condition_code": "J44.1",
                "condition_name": "COPD",
                "probability": 0.16,
                "confidence_interval": (0.04, 0.32),
            },
        ]],
    ]

    cli._get_diagnosis_discriminator.cache_clear()
    cli._get_complaint_generator.cache_clear()

    with (
        patch.object(cli, "create_enhanced_bayesian_network", return_value=FakeNetwork()),
        patch.object(cli, "_get_complaint_generator", return_value=FakeComplaintGenerator()),
        patch.object(cli, "_get_diagnosis_discriminator", return_value=discriminator),
        patch.object(cli, "QuestionGenerator", StubChallengeQuestionGenerator),
        patch.object(cli.random, "choice", side_effect=lambda seq: seq[0]),
    ):
        cli.challenge_command(args)

    output = capsys.readouterr().out

    assert "Top predictions:" in output
    assert "Info sheet preview" in output
    assert "Supporting symptoms:" in output
    assert discriminator.predict_diagnosis.call_count >= 2
