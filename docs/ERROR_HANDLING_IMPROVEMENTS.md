# Error Handling Improvements in _load_llm Methods

## Summary

Improved error handling in the `_load_llm` methods of `phaita/models/generator.py` and `phaita/models/question_generator.py` by implementing more specific exception catching, detailed logging, and clear error messages.

## Changes Made

### 1. Added Specific Exception Handling

**Before (generator.py):**
```python
except ModelDownloadError as e:
    # specific handling
except Exception as e:  # Broad catch-all
    # generic handling
```

**After (both files):**
```python
except ModelDownloadError as e:
    logger.error(f"Model download failed for {model_name}: {type(e).__name__}: {e}")
    # specific handling
except FileNotFoundError as e:
    logger.error(f"Model files not found for {model_name}: {e}")
    # specific handling with helpful message
except (OSError, ValueError, HTTPError) as e:
    logger.error(f"Model loading failed for {model_name}: {type(e).__name__}: {e}")
    # specific handling
```

### 2. Added Logging

- Imported `logging` module in both files
- Created module-level logger: `logger = logging.getLogger(__name__)`
- Added `logger.error()` calls before raising RuntimeError
- Error logs include:
  - Exception type name
  - Model name that failed to load
  - Original error message

### 3. Improved Error Messages

- Error messages now include the specific exception type name (e.g., "OSError:", "ValueError:")
- FileNotFoundError has a special message suggesting the model may not have been downloaded
- All errors preserve the original exception chain with `raise ... from e`

### 4. Added HTTPError Import

Added `from requests.exceptions import HTTPError` to `generator.py` to handle HTTP-related errors during model downloads.

## Exception Types Handled

1. **ModelDownloadError**: Custom exception raised by `model_loader.py` when all download retries fail
2. **FileNotFoundError**: Raised when model files are not found in cache or filesystem
3. **OSError**: Covers disk I/O errors, network errors, permission errors
4. **ValueError**: Raised for invalid model configurations or parameters
5. **HTTPError**: Raised for HTTP-related errors during API calls to HuggingFace Hub

## Benefits

### 1. Better Debugging

With specific exception catching and logging, developers can:
- Quickly identify the root cause of failures
- Distinguish between network issues, missing files, and configuration errors
- See detailed error information in logs even if exceptions are caught elsewhere

### 2. More Resilient Code

- Specific exception handling prevents masking unexpected errors
- FileNotFoundError handling provides actionable guidance
- Error messages guide users toward solutions

### 3. Production-Ready

- Structured logging makes monitoring easier
- Exception type names in error messages help with log parsing
- Original exception chain preserved for full stack traces

## Testing

### New Tests (`tests/test_error_handling.py`)

Created comprehensive test suite with 9 tests covering:
- ModelDownloadError handling in both classes
- FileNotFoundError handling with helpful messages
- OSError and ValueError catching
- Logging behavior verification
- Error message content validation

All tests pass successfully.

### Existing Tests

Verified that changes don't break existing functionality:
- ✅ `test_basic.py`: 4/4 tests passed
- ✅ `test_model_loader.py`: 13/13 tests passed
- ✅ `test_error_handling.py`: 9/9 tests passed
- ✅ `test_enhanced_bayesian.py`: All tests passed
- ✅ `test_template_diversity.py`: 6/6 tests passed

### Demo Script (`demos/demo_error_handling.py`)

Created demonstration script showing:
- How each exception type is caught and handled
- Error logging in action
- Clear error messages with troubleshooting information
- Summary of all improvements

## Retry Mechanism

Note: The retry mechanism with exponential backoff is already implemented in `phaita/utils/model_loader.py`:
- Retries up to 3 times by default
- Exponential backoff: 10s, 20s, 40s between retries
- Catches `(ConnectionError, TimeoutError, OSError)`
- Provides detailed error messages after all retries exhausted

The improvements in `_load_llm` complement this existing retry logic by providing better error reporting when all retries fail.

## Files Modified

1. `phaita/models/generator.py`: Updated error handling in `ComplaintGenerator._load_llm()`
2. `phaita/models/question_generator.py`: Updated error handling in `QuestionGenerator._load_llm()`
3. `tests/test_error_handling.py`: New test suite for error handling
4. `demos/demo_error_handling.py`: New demonstration script

## Example Error Messages

### FileNotFoundError
```
Failed to load model mistralai/Mistral-7B-Instruct-v0.2. 
Model files not found: Model file 'pytorch_model.bin' not found in cache
Requirements:
- transformers==4.46.0
- bitsandbytes==0.44.1 (for 4-bit quantization)
- torch==2.5.1
- CUDA GPU with 4GB+ VRAM recommended (CPU mode available with use_4bit=False)
- Internet connection to download model from HuggingFace Hub
The model may not have been downloaded or the model name is incorrect.
```

### OSError
```
Failed to load model mistralai/Mistral-7B-Instruct-v0.2. 
Error: OSError: Disk I/O error while reading model
Requirements:
- transformers==4.46.0
...
```

## Backward Compatibility

All changes are backward compatible:
- Method signatures unchanged
- Error types remain `RuntimeError` as before
- Existing code using these classes will continue to work
- Only improvement is in error message clarity and logging
