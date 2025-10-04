from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phaita.models.question_generator import QuestionGenerator
from phaita.utils.model_loader import ModelDownloadError


def test_question_generator_download_error_message(monkeypatch):
    """Known download failures surface a helpful runtime error."""

    def _raise_download_error(*args, **kwargs):
        raise ModelDownloadError("network unreachable")

    monkeypatch.setattr(
        "phaita.models.question_generator.load_model_and_tokenizer",
        _raise_download_error,
    )

    with pytest.raises(RuntimeError) as exc_info:
        QuestionGenerator(model_name="fake/model", use_4bit=False)

    message = str(exc_info.value)
    assert "Failed to load model fake/model" in message
    assert "network unreachable" in message


def test_question_generator_unexpected_error_propagates(monkeypatch):
    """Unexpected errors are not masked by the runtime wrapper."""

    class UnexpectedError(RuntimeError):
        pass

    def _raise_unexpected_error(*args, **kwargs):
        raise UnexpectedError("boom")

    monkeypatch.setattr(
        "phaita.models.question_generator.load_model_and_tokenizer",
        _raise_unexpected_error,
    )

    with pytest.raises(UnexpectedError):
        QuestionGenerator(model_name="fake/model", use_4bit=False)
