from argparse import Namespace
from unittest.mock import MagicMock, patch

import cli


def test_diagnose_command_invokes_predict_with_explanation(capsys):
    args = Namespace(complaint="I can't breathe", detailed=True, interactive=False)

    explanation = {
        "condition_code": "J45.9",
        "condition_name": "Asthma",
        "confidence": 0.91,
        "matched_keywords": ["breathe"],
        "reasoning": "Stub reasoning",
        "top_3_predictions": [("J45.9", 0.91), ("J44.1", 0.05)],
    }

    discriminator = MagicMock()
    discriminator.predict_with_explanation.return_value = ("J45.9", 0.91, explanation)

    with patch.object(cli, "_get_diagnosis_discriminator", return_value=discriminator):
        cli.diagnose_command(args)

    output = capsys.readouterr().out
    assert "Primary Diagnosis: Asthma (J45.9)" in output
    assert "Confidence: 0.910" in output
    assert "Differential Ranking" in output
    discriminator.predict_with_explanation.assert_called_once_with("I can't breathe")


def test_challenge_command_uses_real_predictions(capsys):
    args = Namespace(rare_cases=1, atypical_cases=0, show_failures=True, verbose=True)

    class FakeNetwork:
        def sample_symptoms(self, condition_code, **kwargs):
            metadata = {"presentation_type": "rare", "case_name": f"Rare {condition_code}"}
            metadata.update(kwargs)
            return ["cough"], metadata

    class FakeComplaintGenerator:
        def generate_complaint(self, symptoms, condition_code):
            return f"complaint for {condition_code}"

    discriminator = MagicMock()
    discriminator.predict_diagnosis.side_effect = [
        [[{"condition_code": "J45.9", "probability": 0.9}]],
        [[{"condition_code": "J18.9", "probability": 0.6}]],
        [[{"condition_code": "J44.1", "probability": 0.7}]],
        [[{"condition_code": "J18.9", "probability": 0.4}]],
    ]

    with (
        patch.object(cli, "create_enhanced_bayesian_network", return_value=FakeNetwork()),
        patch.object(cli, "_get_complaint_generator", return_value=FakeComplaintGenerator()),
        patch.object(cli, "_get_diagnosis_discriminator", return_value=discriminator),
        patch.object(cli.random, "choice", side_effect=lambda seq: seq[0]),
    ):
        cli.challenge_command(args)

    output = capsys.readouterr().out
    assert "Top predictions:" in output
    assert "Model Ranking" in output
    assert discriminator.predict_diagnosis.call_count == 4
    assert output.count("Case") >= 1
