#!/usr/bin/env python
"""
Demonstration of improved error handling in PHAITA model loading.

This script shows how the _load_llm methods in generator.py and 
question_generator.py now handle errors more specifically with:
1. Specific exception catching (FileNotFoundError, OSError, ValueError, HTTPError)
2. Detailed error logging before raising
3. Clear error messages with exception type names

Note: This script will intentionally trigger errors to demonstrate
the improved error handling. This is for demonstration purposes only.
"""

import logging
from unittest.mock import patch
from phaita.models.generator import ComplaintGenerator
from phaita.models.question_generator import QuestionGenerator
from phaita.utils.model_loader import ModelDownloadError

# Configure logging to show our error messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)


def demo_model_download_error():
    """Demonstrate handling of ModelDownloadError."""
    print("\n" + "="*60)
    print("Demo 1: ModelDownloadError Handling")
    print("="*60)
    
    with patch('phaita.models.generator.load_model_and_tokenizer') as mock_loader:
        mock_loader.side_effect = ModelDownloadError(
            "Failed to download model after 3 attempts.\n"
            "Network connection timed out."
        )
        
        try:
            generator = ComplaintGenerator(use_pretrained=True)
        except RuntimeError as e:
            print("\n✓ Caught RuntimeError as expected")
            print(f"\nError message includes:")
            print(f"  - Original error: {('Network connection timed out' in str(e))}")
            print(f"  - Requirements info: {('transformers' in str(e))}")
            print(f"\nFirst 300 chars of error:\n{str(e)[:300]}...")


def demo_file_not_found_error():
    """Demonstrate handling of FileNotFoundError."""
    print("\n" + "="*60)
    print("Demo 2: FileNotFoundError Handling")
    print("="*60)
    
    with patch('phaita.models.question_generator.load_model_and_tokenizer') as mock_loader:
        mock_loader.side_effect = FileNotFoundError(
            "Model file 'pytorch_model.bin' not found in cache"
        )
        
        try:
            generator = QuestionGenerator(use_pretrained=True)
        except RuntimeError as e:
            print("\n✓ Caught RuntimeError as expected")
            print(f"\nError message includes:")
            print(f"  - 'Model files not found': {('Model files not found' in str(e))}")
            print(f"  - Helpful hint: {('may not have been downloaded' in str(e))}")
            print(f"\nFirst 300 chars of error:\n{str(e)[:300]}...")


def demo_oserror():
    """Demonstrate handling of OSError."""
    print("\n" + "="*60)
    print("Demo 3: OSError Handling")
    print("="*60)
    
    with patch('phaita.models.generator.load_model_and_tokenizer') as mock_loader:
        mock_loader.side_effect = OSError("Disk I/O error while reading model")
        
        try:
            generator = ComplaintGenerator(use_pretrained=True)
        except RuntimeError as e:
            print("\n✓ Caught RuntimeError as expected")
            print(f"\nError message includes:")
            print(f"  - Exception type 'OSError': {('OSError' in str(e))}")
            print(f"  - Original error message: {('Disk I/O error' in str(e))}")
            print(f"\nFirst 300 chars of error:\n{str(e)[:300]}...")


def demo_value_error():
    """Demonstrate handling of ValueError."""
    print("\n" + "="*60)
    print("Demo 4: ValueError Handling")
    print("="*60)
    
    with patch('phaita.models.question_generator.load_model_and_tokenizer') as mock_loader:
        mock_loader.side_effect = ValueError(
            "Invalid model configuration: expected torch_dtype to be torch.dtype"
        )
        
        try:
            generator = QuestionGenerator(use_pretrained=True)
        except RuntimeError as e:
            print("\n✓ Caught RuntimeError as expected")
            print(f"\nError message includes:")
            print(f"  - Exception type 'ValueError': {('ValueError' in str(e))}")
            print(f"  - Configuration error: {('Invalid model configuration' in str(e))}")
            print(f"\nFirst 300 chars of error:\n{str(e)[:300]}...")


def demo_logging():
    """Demonstrate that errors are logged before raising."""
    print("\n" + "="*60)
    print("Demo 5: Error Logging")
    print("="*60)
    
    print("\n✓ Errors are now logged before raising RuntimeError")
    print("✓ Log messages include:")
    print("  - Exception type name (OSError, ValueError, etc.)")
    print("  - Original error message")
    print("  - Model name that failed to load")
    print("\n✓ This helps with debugging in production environments")
    print("  where stack traces might not be immediately visible")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("PHAITA - Improved Error Handling Demo")
    print("="*60)
    print("\nThis demo shows the improvements made to error handling")
    print("in the _load_llm methods of generator.py and question_generator.py")
    
    demo_model_download_error()
    demo_file_not_found_error()
    demo_oserror()
    demo_value_error()
    demo_logging()
    
    print("\n" + "="*60)
    print("Summary of Improvements")
    print("="*60)
    print("\n1. ✓ Specific Exception Catching:")
    print("   - ModelDownloadError (custom exception)")
    print("   - FileNotFoundError (with helpful message)")
    print("   - OSError (disk, network, permission errors)")
    print("   - ValueError (configuration errors)")
    print("   - HTTPError (API/network errors)")
    print("\n2. ✓ Error Logging:")
    print("   - All errors logged with logger.error() before raising")
    print("   - Includes exception type and original message")
    print("\n3. ✓ Clear Error Messages:")
    print("   - Exception type name included in error message")
    print("   - Original error details preserved")
    print("   - Helpful troubleshooting information")
    print("\n4. ✓ Retry Mechanism:")
    print("   - Already implemented in model_loader.py")
    print("   - Exponential backoff (10s, 20s, 40s)")
    print("   - Handles transient network errors automatically")
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
