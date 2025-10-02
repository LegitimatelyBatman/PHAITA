from argparse import Namespace
from unittest.mock import MagicMock, patch

import cli
from phaita.generation.patient_agent import PatientPresentation, VocabularyProfile


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
        def generate_complaint(
            self,
            condition_code=None,
            presentation=None,
            symptoms=None,
            **kwargs,
        ):
            condition = condition_code or "J45.9"
            symptom_list = symptoms or []
            if presentation is None:
                presentation = PatientPresentation(
                    condition_code=condition,
                    symptoms=list(symptom_list),
                    symptom_probabilities={symptom: 0.8 for symptom in symptom_list},
                    misdescription_weights={symptom: 0.2 for symptom in symptom_list},
                    vocabulary_profile=VocabularyProfile.default_for(symptom_list),
                )
            presentation.complaint_text = f"complaint for {condition}"
            return presentation

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
