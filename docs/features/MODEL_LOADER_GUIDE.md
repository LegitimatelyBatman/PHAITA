# Model Loader - Retry Mechanisms Guide

## Overview

The PHAITA model loader provides robust downloading with automatic retries, exponential backoff, and offline mode support. This ensures reliable model loading even with unstable network connections.

## Features

### ✅ Automatic Retry Logic
- Retries downloads on network failures (ConnectionError, TimeoutError, OSError)
- Default: 3 retry attempts per model
- Exponential backoff: 10s → 20s → 40s between retries

### ✅ Partial Download Recovery
- Uses `resume_download=True` by default
- Recovers partial downloads from HuggingFace cache
- Checks `~/.cache/huggingface/hub` for existing files

### ✅ Offline Mode
- Supports `allow_offline=True` to use cached models only
- Clear error messages when models not cached
- Provides `huggingface-cli download` commands for manual downloads

### ✅ Clear Error Messages
- Troubleshooting steps included in errors
- Shows attempted models and last error
- Includes cache location and manual download instructions

## Usage

### Basic Usage

```python
from phaita.utils.model_loader import robust_model_download, ModelDownloadError

# Download a tokenizer with retry support
tokenizer = robust_model_download(
    model_name="microsoft/deberta-v3-base",
    model_type="tokenizer",
    max_retries=3,
    timeout=300
)

# Download a model
model = robust_model_download(
    model_name="microsoft/deberta-v3-base",
    model_type="auto",  # or "causal_lm" for language models
    max_retries=3,
    timeout=300
)
```

### Load Model and Tokenizer Together

```python
from phaita.utils.model_loader import load_model_and_tokenizer

# Load both in one call
model, tokenizer = load_model_and_tokenizer(
    model_name="microsoft/deberta-v3-base",
    model_type="auto",
    max_retries=3,
    timeout=300
)
```

### Offline Mode

```python
# Only use cached models (no network access)
try:
    model = robust_model_download(
        model_name="gpt2",
        model_type="causal_lm",
        allow_offline=True
    )
except ModelDownloadError as e:
    print(f"Model not cached: {e}")
    # Error message includes instructions for manual download
```

### Custom Configuration

```python
# Adjust retry parameters
model = robust_model_download(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    model_type="causal_lm",
    max_retries=5,           # More retries for large models
    timeout=600,             # Longer timeout (10 minutes)
    device_map="auto",       # Pass through to from_pretrained()
    torch_dtype="float16"
)
```

## Parameters

### `robust_model_download()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | required | HuggingFace model name (e.g., "microsoft/deberta-v3-base") |
| `model_type` | str | "auto" | Model type: "auto", "causal_lm", or "tokenizer" |
| `max_retries` | int | 3 | Maximum number of retry attempts |
| `timeout` | int | 300 | Timeout in seconds for each attempt |
| `allow_offline` | bool | False | Only use cached models (no download) |
| `token` | str | None | HuggingFace authentication token |
| `**kwargs` | Any | - | Additional arguments passed to `from_pretrained()` |

## Integration with PHAITA

The robust model loader is automatically used by:

- **DiagnosisDiscriminator**: DeBERTa text encoder loading
- **ComplaintGenerator**: Mistral-7B language model loading
- **QuestionGenerator**: Mistral-7B for question generation
- **RealismScorer**: BERT/BioBERT and GPT-2 model loading

All models now benefit from:
- ✅ Automatic retries on network failures
- ✅ Exponential backoff between retries
- ✅ Resume partial downloads
- ✅ Clear, actionable error messages

## Error Handling

### Network Failures

```python
try:
    model = robust_model_download("microsoft/deberta-v3-base", model_type="auto")
except ModelDownloadError as e:
    print(f"Download failed after all retries: {e}")
    # Error includes troubleshooting steps
```

### Offline Mode Errors

```python
try:
    model = robust_model_download(
        "microsoft/deberta-v3-base",
        model_type="auto",
        allow_offline=True
    )
except ModelDownloadError as e:
    # Error message includes:
    # - Cache directory location
    # - Manual download command
    # - Instructions to disable offline mode
    print(e)
```

## Retry Behavior

The loader uses exponential backoff:

1. **Attempt 1**: Initial download
2. **Attempt 1 fails** → wait 10 seconds
3. **Attempt 2**: Retry download
4. **Attempt 2 fails** → wait 20 seconds
5. **Attempt 3**: Final retry
6. **Attempt 3 fails** → raise `ModelDownloadError`

Wait times: `2^attempt * 10` seconds (10s, 20s, 40s, 80s...)

## Cache Management

### Check if Model is Cached

```python
from phaita.utils.model_loader import _check_cached_model

if _check_cached_model("microsoft/deberta-v3-base"):
    print("Model is cached")
else:
    print("Model needs to be downloaded")
```

### Cache Location

Default: `~/.cache/huggingface/hub`

Override with environment variables:
- `HF_HOME`
- `TRANSFORMERS_CACHE`

### Manual Download

If automatic download fails:

```bash
# Using HuggingFace CLI
huggingface-cli download microsoft/deberta-v3-base

# Or using Python
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/deberta-v3-base')"
```

## Testing

Run the comprehensive test suite:

```bash
python test_model_loader.py
```

Tests include:
- Retry logic with mocked failures
- Exponential backoff timing
- Offline mode behavior
- Error message clarity
- Tokenizer and model loading
- Resume download support

Run the demo:

```bash
python demo_model_loader.py
```

## Troubleshooting

### "Failed to download model after N attempts"

1. Check internet connection
2. Verify model name is correct
3. Try manual download with `huggingface-cli`
4. Check HuggingFace status: https://status.huggingface.co/
5. Increase `max_retries` or `timeout`

### "Model not found in cache and offline mode is enabled"

1. Download model manually first
2. Check cache directory exists: `~/.cache/huggingface/hub`
3. Disable offline mode if network is available

### Slow Downloads

1. Use `timeout` parameter for long-running downloads
2. Large models (Mistral-7B) may take 5-10 minutes
3. Consider downloading manually first for large models

## Examples

### Example 1: Discriminator Loading

```python
from phaita.models import DiagnosisDiscriminator

# Automatically uses robust_model_download with retries
discriminator = DiagnosisDiscriminator(
    model_name="microsoft/deberta-v3-base",
    use_pretrained=True
)
```

### Example 2: Generator Loading

```python
from phaita.models import ComplaintGenerator

# Automatically uses robust_model_download with retries
generator = ComplaintGenerator(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    use_pretrained=True,
    use_4bit=True
)
```

### Example 3: Custom Retry Configuration

```python
from phaita.utils.model_loader import robust_model_download

# For large models on slow connections
model = robust_model_download(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    model_type="causal_lm",
    max_retries=5,      # More retries
    timeout=900,        # 15 minute timeout
    use_4bit=True,
    device_map="auto"
)
```

## See Also

- `phaita/utils/model_loader.py` - Implementation
- `test_model_loader.py` - Comprehensive tests
- `demo_model_loader.py` - Interactive demonstration
- `DEEP_LEARNING_GUIDE.md` - Deep learning setup guide
