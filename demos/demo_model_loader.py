#!/usr/bin/env python3
"""
Demo script showing the robust model loader in action.
This demonstrates the retry mechanism, offline mode, and error handling.
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phaita.utils.model_loader import (
    robust_model_download,
    load_model_and_tokenizer,
    ModelDownloadError,
    _check_cached_model,
)


def demo_cache_check():
    """Demonstrate checking if models are cached."""
    print("=" * 60)
    print("Demo 1: Checking Model Cache")
    print("=" * 60)
    
    test_models = [
        "microsoft/deberta-v3-base",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "bert-base-uncased",
        "gpt2",
    ]
    
    for model_name in test_models:
        is_cached = _check_cached_model(model_name)
        status = "✓ CACHED" if is_cached else "✗ NOT CACHED"
        print(f"{status}: {model_name}")
    
    print()


def demo_offline_mode():
    """Demonstrate offline mode behavior."""
    print("=" * 60)
    print("Demo 2: Offline Mode (uses cached models only)")
    print("=" * 60)
    
    # Try to load a commonly cached model (gpt2)
    try:
        print("Attempting to load 'gpt2' in offline mode...")
        tokenizer = robust_model_download(
            model_name="gpt2",
            model_type="tokenizer",
            allow_offline=True,
            max_retries=1
        )
        print("✓ Successfully loaded gpt2 tokenizer from cache")
    except ModelDownloadError as e:
        print(f"✗ Failed (expected if not cached): {str(e)[:100]}...")
    
    print()


def demo_retry_parameters():
    """Demonstrate retry configuration."""
    print("=" * 60)
    print("Demo 3: Retry Configuration Options")
    print("=" * 60)
    
    print("Available parameters for robust_model_download():")
    print("  - model_name: Name of the model (e.g., 'microsoft/deberta-v3-base')")
    print("  - model_type: 'auto', 'causal_lm', or 'tokenizer'")
    print("  - max_retries: Number of retry attempts (default: 3)")
    print("  - timeout: Timeout in seconds (default: 300)")
    print("  - allow_offline: Only use cached models (default: False)")
    print("  - **kwargs: Additional args passed to from_pretrained()")
    print()
    print("Retry backoff: 2^attempt * 10 seconds")
    print("  Attempt 1 fails → wait 10s")
    print("  Attempt 2 fails → wait 20s")
    print("  Attempt 3 fails → wait 40s")
    print()


def demo_error_messages():
    """Demonstrate error message clarity."""
    print("=" * 60)
    print("Demo 4: Error Message Examples")
    print("=" * 60)
    
    print("Example error when model not found in offline mode:")
    print("-" * 60)
    try:
        robust_model_download(
            model_name="nonexistent/fake-model-12345",
            model_type="tokenizer",
            allow_offline=True,
            max_retries=1
        )
    except ModelDownloadError as e:
        print(str(e))
    
    print()


def demo_integrated_loading():
    """Show how the new loader integrates with existing code."""
    print("=" * 60)
    print("Demo 5: Integration with PHAITA Models")
    print("=" * 60)
    
    print("The robust model loader is now used by:")
    print("  ✓ DiagnosisDiscriminator (discriminator.py)")
    print("  ✓ ComplaintGenerator (generator.py)")
    print("  ✓ QuestionGenerator (question_generator.py)")
    print("  ✓ RealismScorer (realism_scorer.py)")
    print()
    print("All model loading now includes:")
    print("  - Automatic retry on network failures")
    print("  - Exponential backoff (10s, 20s, 40s...)")
    print("  - Resume partial downloads")
    print("  - Clear error messages with troubleshooting tips")
    print("  - Offline mode support")
    print()


def main():
    """Run all demos."""
    print("\n🔧 Robust Model Loader Demo\n")
    
    demo_cache_check()
    demo_offline_mode()
    demo_retry_parameters()
    demo_error_messages()
    demo_integrated_loading()
    
    print("=" * 60)
    print("✓ Demo complete!")
    print("=" * 60)
    print()
    print("For more information:")
    print("  - See test_model_loader.py for comprehensive tests")
    print("  - Check phaita/utils/model_loader.py for implementation")
    print("  - Run 'python test_model_loader.py' to test retry logic")
    print()


if __name__ == "__main__":
    main()
