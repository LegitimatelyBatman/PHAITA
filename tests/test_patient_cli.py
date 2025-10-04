import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_cli import ConversationIO, run_interview


class ScriptedIO(ConversationIO):
    """Simple scripted I/O helper for exercising the CLI logic."""

    def __init__(self, prompts):
        self._prompts = iter(prompts)
        self.outputs = []

    def display(self, text: str) -> None:
        self.outputs.append(text)

    def prompt(self):
        try:
            return next(self._prompts)
        except StopIteration:
            return None


def test_patient_cli_respects_vocabulary_limits_and_reveals_truth():
    io = ScriptedIO([
        "How long have you felt this way?",
        "diagnose asthma",
    ])

    result = run_interview(
        io,
        condition_code="J45.9",
        seed=0,
        max_terms=2,
        strategy="detailed",
    )

    # Ensure the reveal happened and contained the hidden metadata.
    assert any("Ground Truth Reveal" in line for line in io.outputs)
    assert any("Condition Code: J45.9" in line for line in io.outputs)

    # Each follow-up exchange should include metadata about symptom mentions
    # with a count that respects the vocabulary constraint.
    max_terms = result.presentation.vocabulary_profile.max_terms_per_response
    for exchange in result.presentation.follow_up_history:
        mentions = exchange.get("symptom_mentions")
        assert mentions is not None
        assert len(mentions) <= max_terms

    assert result.exit_reason == "diagnosis"
    assert result.final_diagnosis == "asthma"


def test_patient_cli_exit_command_triggers_reveal():
    io = ScriptedIO(["exit"])

    result = run_interview(
        io,
        condition_code="J45.9",
        seed=1,
    )

    assert result.exit_reason == "exit"
    assert result.final_diagnosis is None
    assert any("Final Diagnosis Received: (none)" in line for line in io.outputs)
    assert any("True Symptoms:" in line for line in io.outputs)
