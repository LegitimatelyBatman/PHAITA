"""
Robust model loader with retry mechanisms and offline fallback.
Handles network timeouts, partial downloads, and provides clear error messages.
"""

import logging
import time
import os
from pathlib import Path
from typing import Optional, Union, Tuple, Any
import hashlib

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForCausalLM,
        PreTrainedTokenizer,
        PreTrainedModel,
    )
    from transformers.utils import TRANSFORMERS_CACHE
except ImportError as e:
    raise ImportError(
        "transformers is required for model_loader. "
        "Install with: pip install transformers==4.46.0"
    ) from e

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None


logger = logging.getLogger(__name__)


class ModelDownloadError(Exception):
    """Raised when model download fails after all retries."""
    pass


def _get_cache_dir() -> Path:
    """Get the HuggingFace cache directory."""
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    if cache_dir:
        return Path(cache_dir)
    
    # Default cache location
    home = Path.home()
    return home / ".cache" / "huggingface" / "hub"


def _check_cached_model(model_name: str) -> bool:
    """
    Check if a model is available in the local cache.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model is cached, False otherwise
    """
    cache_dir = _get_cache_dir()
    
    if not cache_dir.exists():
        return False
    
    # HuggingFace stores models with names like "models--microsoft--deberta-v3-base"
    safe_name = model_name.replace("/", "--")
    model_dirs = [
        cache_dir / f"models--{safe_name}",
        cache_dir / safe_name,
    ]
    
    for model_dir in model_dirs:
        if model_dir.exists() and any(model_dir.iterdir()):
            logger.info(f"Found cached model at: {model_dir}")
            return True
    
    return False


def robust_model_download(
    model_name: str,
    model_type: str = "auto",
    max_retries: int = 3,
    timeout: int = 300,
    allow_offline: bool = False,
    token: Optional[str] = None,
    **kwargs
) -> Union[PreTrainedModel, PreTrainedTokenizer]:
    """
    Download a model with retry logic and exponential backoff.
    
    Args:
        model_name: Name of the model to download (e.g., "microsoft/deberta-v3-base")
        model_type: Type of model to load:
            - "auto": AutoModel
            - "causal_lm": AutoModelForCausalLM
            - "tokenizer": AutoTokenizer
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each download attempt
        allow_offline: If True, only use cached models (no download)
        token: Optional HuggingFace authentication token
        **kwargs: Additional arguments to pass to from_pretrained()
        
    Returns:
        Loaded model or tokenizer
        
    Raises:
        ModelDownloadError: If download fails after all retries
        ValueError: If model_type is invalid
    """
    # Validate model_type
    valid_types = ["auto", "causal_lm", "tokenizer"]
    if model_type not in valid_types:
        raise ValueError(f"model_type must be one of {valid_types}, got: {model_type}")
    
    # Check if model is already cached
    is_cached = _check_cached_model(model_name)
    
    if allow_offline and not is_cached:
        raise ModelDownloadError(
            f"Model '{model_name}' not found in cache and offline mode is enabled.\n"
            f"Cache directory: {_get_cache_dir()}\n"
            f"To download manually, run:\n"
            f"  huggingface-cli download {model_name}\n"
            f"Or disable offline mode."
        )
    
    # Select the appropriate loading function
    if model_type == "tokenizer":
        load_fn = AutoTokenizer.from_pretrained
    elif model_type == "causal_lm":
        load_fn = AutoModelForCausalLM.from_pretrained
    else:  # "auto"
        load_fn = AutoModel.from_pretrained
    
    # Prepare loading arguments
    load_kwargs = {
        "resume_download": True,
        **kwargs
    }
    
    # Add auth token if provided
    if token:
        load_kwargs["token"] = token
    
    # If offline mode or cached, try to load from cache first
    if allow_offline or is_cached:
        load_kwargs["local_files_only"] = True
        try:
            logger.info(f"Loading {model_name} from cache (offline mode)...")
            model = load_fn(model_name, **load_kwargs)
            logger.info(f"✓ Loaded {model_name} from cache")
            return model
        except Exception as e:
            if allow_offline:
                raise ModelDownloadError(
                    f"Failed to load model '{model_name}' from cache in offline mode.\n"
                    f"Error: {e}\n"
                    f"The cached model may be corrupted or incomplete."
                ) from e
            else:
                logger.warning(f"Failed to load from cache, will try downloading: {e}")
                load_kwargs["local_files_only"] = False
    
    # Retry logic with exponential backoff
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} for {model_name}...")
            else:
                logger.info(f"Downloading {model_name}...")
            
            # Load the model with resume support
            model = load_fn(model_name, **load_kwargs)
            
            logger.info(f"✓ Successfully loaded {model_name}")
            return model
            
        except (ConnectionError, TimeoutError, OSError) as e:
            last_error = e
            
            if attempt < max_retries - 1:
                # Exponential backoff: 10s, 20s, 40s...
                wait_time = 2 ** attempt * 10
                logger.warning(
                    f"Download failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}\n"
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                # Final attempt failed
                logger.error(f"All {max_retries} download attempts failed for {model_name}")
    
    # All retries exhausted
    error_msg = (
        f"Failed to download model '{model_name}' after {max_retries} attempts.\n"
        f"Last error: {type(last_error).__name__}: {last_error}\n"
        f"\nTroubleshooting:\n"
        f"1. Check your internet connection\n"
        f"2. Verify the model name is correct: {model_name}\n"
        f"3. Try downloading manually:\n"
        f"   huggingface-cli download {model_name}\n"
        f"4. Check HuggingFace status: https://status.huggingface.co/\n"
    )
    
    if is_cached:
        error_msg += (
            f"\nNote: A cached version was found but may be incomplete.\n"
            f"Cache location: {_get_cache_dir()}\n"
            f"Try clearing the cache and re-downloading."
        )
    
    raise ModelDownloadError(error_msg) from last_error


def load_model_and_tokenizer(
    model_name: str,
    model_type: str = "auto",
    max_retries: int = 3,
    timeout: int = 300,
    allow_offline: bool = False,
    token: Optional[str] = None,
    **model_kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load both model and tokenizer with retry logic.
    
    Args:
        model_name: Name of the model to download
        model_type: Type of model ("auto" or "causal_lm")
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for each download attempt
        allow_offline: If True, only use cached models
        token: Optional HuggingFace authentication token
        **model_kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        ModelDownloadError: If download fails
    """
    # Load tokenizer first (smaller, faster)
    tokenizer = robust_model_download(
        model_name=model_name,
        model_type="tokenizer",
        max_retries=max_retries,
        timeout=timeout,
        allow_offline=allow_offline,
        token=token,
    )
    
    # Then load the model
    model = robust_model_download(
        model_name=model_name,
        model_type=model_type,
        max_retries=max_retries,
        timeout=timeout,
        allow_offline=allow_offline,
        token=token,
        **model_kwargs
    )
    
    return model, tokenizer


# Convenience function for legacy compatibility
def load_pretrained_model(
    model_name: str,
    model_class: str = "AutoModel",
    **kwargs
) -> Union[PreTrainedModel, PreTrainedTokenizer]:
    """
    Legacy interface for loading models with retry support.
    
    Args:
        model_name: Model name
        model_class: Class name ("AutoModel", "AutoModelForCausalLM", "AutoTokenizer")
        **kwargs: Additional arguments
        
    Returns:
        Loaded model or tokenizer
    """
    type_map = {
        "AutoModel": "auto",
        "AutoModelForCausalLM": "causal_lm",
        "AutoTokenizer": "tokenizer",
    }
    
    model_type = type_map.get(model_class, "auto")
    return robust_model_download(model_name, model_type=model_type, **kwargs)
