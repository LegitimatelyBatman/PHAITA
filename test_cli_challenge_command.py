from argparse import Namespace
from types import SimpleNamespace

import cli


class _StubNetwork:
    def __init__(self):
        self.calls = []

    def sample_symptoms(self, condition_code, **kwargs):
        if "comorbidities" in kwargs:
            raise AssertionError("comorbidities argument should not be passed")

        self.calls.append((condition_code, dict(kwargs)))

        metadata = {
            "presentation_type": "rare" if kwargs.get("include_rare") else "standard",
            "case_name": f"Case for {condition_code}",
        }
        metadata.setdefault("age_group", kwargs.get("age_group", "adult"))
        metadata.setdefault("severity", kwargs.get("severity", "moderate"))

        return [f"{condition_code}_symptom"], metadata


class _StubComplaintGenerator:
    def generate_complaint(self, condition_code, symptoms):
        return SimpleNamespace(complaint_text=f"Complaint for {condition_code}")


class _StubDiscriminator:
    def predict_diagnosis(self, transcripts, top_k=3):
        return [[{"condition_code": "J45.9", "probability": 0.5}]]


class _StubEngine:
    def __init__(self, *args, **kwargs):
        self._symptoms = []

    def add_symptoms(self, symptoms):
        self._symptoms.extend(symptoms)

    def update_differential(self, ranked):
        return None

    def should_present_diagnosis(self):
        return True

    def next_prompt(self):
        return None

    def record_response(self, prompt, response, extracted_symptoms):
        return None


def test_challenge_command_handles_comorbidity_cases(monkeypatch, capsys):
    stub_network = _StubNetwork()

    monkeypatch.setattr(cli, "create_enhanced_bayesian_network", lambda: stub_network)
    monkeypatch.setattr(cli, "_get_complaint_generator", lambda: _StubComplaintGenerator())
    monkeypatch.setattr(cli, "_get_diagnosis_discriminator", lambda: _StubDiscriminator())
    monkeypatch.setattr(cli, "ConversationEngine", _StubEngine)
    monkeypatch.setattr(cli, "_build_question_generator", lambda: object())

    args = Namespace(
        rare_cases=1,
        atypical_cases=1,
        show_failures=False,
        verbose=False,
    )

    cli.challenge_command(args)

    assert len(stub_network.calls) >= 3
    assert all("comorbidities" not in kwargs for _, kwargs in stub_network.calls)

