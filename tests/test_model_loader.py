"""
Test suite for robust model loader with retry mechanisms.
Tests retry logic, exponential backoff, offline mode, and error handling.
"""

import time
from unittest.mock import Mock, patch

import pytest


from phaita.utils.model_loader import (  # noqa: E402
    ModelDownloadError,
    load_model_and_tokenizer,
    robust_model_download,
)


def test_retry_logic_with_network_failures():
    """Test that the loader retries on network failures."""
    mock_model = Mock()
    call_count = [0]

    def mock_from_pretrained(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError("Network error")
        return mock_model

    with patch("phaita.utils.model_loader.AutoModel") as mock_auto_model:
        mock_auto_model.from_pretrained = mock_from_pretrained

        result = robust_model_download(
            "test-model",
            model_type="auto",
            max_retries=3,
            timeout=10,
        )

        assert result is mock_model
        assert call_count[0] == 3  # Failed twice, succeeded on third


def test_exponential_backoff_timing():
    """Test that exponential backoff increases wait time correctly."""
    start_times = []

    def mock_from_pretrained(*args, **kwargs):
        start_times.append(time.time())
        if len(start_times) < 3:
            raise ConnectionError("Network error")
        return Mock()

    with patch("phaita.utils.model_loader.AutoModel") as mock_auto_model:
        mock_auto_model.from_pretrained = mock_from_pretrained

        with patch("time.sleep") as mock_sleep:
            robust_model_download(
                "test-model",
                model_type="auto",
                max_retries=3,
                timeout=10,
            )

            assert mock_sleep.call_count == 2
            # First backoff: 2^0 * 10 = 10s
            assert mock_sleep.call_args_list[0][0][0] == 10
            # Second backoff: 2^1 * 10 = 20s
            assert mock_sleep.call_args_list[1][0][0] == 20


def test_offline_mode_uses_cached_models():
    """Test that offline mode only uses cached models."""
    mock_model = Mock()

    with patch("phaita.utils.model_loader._check_cached_model", return_value=True):
        with patch("phaita.utils.model_loader.AutoModel") as mock_auto_model:
            mock_auto_model.from_pretrained = Mock(return_value=mock_model)

            result = robust_model_download(
                "test-model",
                model_type="auto",
                allow_offline=True,
            )

            assert result is mock_model
            call_kwargs = mock_auto_model.from_pretrained.call_args[1]
            assert call_kwargs.get("local_files_only")


def test_offline_mode_fails_without_cache():
    """Test that offline mode raises error when model not cached."""
    with patch("phaita.utils.model_loader._check_cached_model", return_value=False):
        with pytest.raises(ModelDownloadError) as cm:
            robust_model_download(
                "test-model",
                model_type="auto",
                allow_offline=True,
            )

    error_msg = str(cm.value)
    assert "not found in cache" in error_msg
    assert "offline mode is enabled" in error_msg
    assert "huggingface-cli download" in error_msg


def test_error_message_clarity_after_all_retries_fail():
    """Test that error messages are clear when all retries fail."""

    def mock_from_pretrained(*args, **kwargs):
        raise ConnectionError("Connection timeout")

    with patch("phaita.utils.model_loader.AutoModel") as mock_auto_model:
        mock_auto_model.from_pretrained = mock_from_pretrained

        with patch("time.sleep"):
            with pytest.raises(ModelDownloadError) as cm:
                robust_model_download(
                    "microsoft/deberta-v3-base",
                    model_type="auto",
                    max_retries=2,
                    timeout=10,
                )

    error_msg = str(cm.value)
    assert "Failed to download model" in error_msg
    assert "after 2 attempts" in error_msg
    assert "microsoft/deberta-v3-base" in error_msg
    assert "Troubleshooting" in error_msg
    assert "internet connection" in error_msg
    assert "huggingface-cli download" in error_msg


def test_tokenizer_loading():
    """Test that tokenizer can be loaded separately."""
    mock_tokenizer = Mock()

    with patch("phaita.utils.model_loader.AutoTokenizer") as mock_auto_tokenizer:
        mock_auto_tokenizer.from_pretrained = Mock(return_value=mock_tokenizer)

        result = robust_model_download(
            "test-model",
            model_type="tokenizer",
            max_retries=1,
        )

        assert result is mock_tokenizer
        assert mock_auto_tokenizer.from_pretrained.called


def test_causal_lm_loading():
    """Test that causal language models can be loaded."""
    mock_model = Mock()

    with patch("phaita.utils.model_loader.AutoModelForCausalLM") as mock_auto_model:
        mock_auto_model.from_pretrained = Mock(return_value=mock_model)

        result = robust_model_download(
            "test-model",
            model_type="causal_lm",
            max_retries=1,
        )

        assert result is mock_model
        assert mock_auto_model.from_pretrained.called


def test_load_model_and_tokenizer_together():
    """Test loading model and tokenizer together."""
    mock_model = Mock()
    mock_tokenizer = Mock()

    with patch("phaita.utils.model_loader.AutoModel") as mock_auto_model:
        with patch("phaita.utils.model_loader.AutoTokenizer") as mock_auto_tokenizer:
            mock_auto_model.from_pretrained = Mock(return_value=mock_model)
            mock_auto_tokenizer.from_pretrained = Mock(return_value=mock_tokenizer)

            model, tokenizer = load_model_and_tokenizer(
                "test-model",
                model_type="auto",
                max_retries=1,
            )

    assert model is mock_model
    assert tokenizer is mock_tokenizer


def test_resume_download_enabled():
    """Test that resume_download is enabled by default."""
    mock_model = Mock()

    with patch("phaita.utils.model_loader.AutoModel") as mock_auto_model:
        mock_auto_model.from_pretrained = Mock(return_value=mock_model)

        robust_model_download(
            "test-model",
            model_type="auto",
            max_retries=1,
        )

    call_kwargs = mock_auto_model.from_pretrained.call_args[1]
    assert call_kwargs.get("resume_download")


def test_invalid_model_type_raises_error():
    """Test that invalid model_type raises ValueError."""
    with pytest.raises(ValueError) as cm:
        robust_model_download(
            "test-model",
            model_type="invalid_type",
            max_retries=1,
        )

    assert "model_type must be one of" in str(cm.value)


def test_kwargs_passed_to_from_pretrained():
    """Test that additional kwargs are passed through."""
    mock_model = Mock()

    with patch("phaita.utils.model_loader.AutoModel") as mock_auto_model:
        mock_auto_model.from_pretrained = Mock(return_value=mock_model)

        robust_model_download(
            "test-model",
            model_type="auto",
            max_retries=1,
            device_map="auto",
            torch_dtype="float16",
        )

    call_kwargs = mock_auto_model.from_pretrained.call_args[1]
    assert call_kwargs.get("device_map") == "auto"
    assert call_kwargs.get("torch_dtype") == "float16"


def test_token_keyword_forwarded_to_transformers():
    """Token argument should be forwarded using the new keyword."""
    mock_model = Mock()

    with patch("phaita.utils.model_loader.AutoModel") as mock_auto_model:
        mock_auto_model.from_pretrained = Mock(return_value=mock_model)

        robust_model_download(
            "test-model",
            model_type="auto",
            max_retries=1,
            token="hf_secret",
        )

    call_kwargs = mock_auto_model.from_pretrained.call_args[1]
    assert call_kwargs.get("token") == "hf_secret"
    assert "use_auth_token" not in call_kwargs


def test_token_forwarded_for_model_and_tokenizer():
    """Token argument should be forwarded when loading both model and tokenizer."""
    mock_model = Mock()
    mock_tokenizer = Mock()

    with patch("phaita.utils.model_loader.AutoModel") as mock_auto_model, patch(
        "phaita.utils.model_loader.AutoTokenizer"
    ) as mock_auto_tokenizer:
        mock_auto_model.from_pretrained = Mock(return_value=mock_model)
        mock_auto_tokenizer.from_pretrained = Mock(return_value=mock_tokenizer)

        model, tokenizer = load_model_and_tokenizer(
            "test-model",
            model_type="auto",
            max_retries=1,
            token="hf_secret",
        )

    assert model is mock_model
    assert tokenizer is mock_tokenizer

    model_kwargs = mock_auto_model.from_pretrained.call_args[1]
    tokenizer_kwargs = mock_auto_tokenizer.from_pretrained.call_args[1]

    assert model_kwargs.get("token") == "hf_secret"
    assert tokenizer_kwargs.get("token") == "hf_secret"
    assert "use_auth_token" not in model_kwargs
    assert "use_auth_token" not in tokenizer_kwargs
