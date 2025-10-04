"""
Test suite for robust model loader with retry mechanisms.
Tests retry logic, exponential backoff, offline mode, and error handling.
"""

import sys
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phaita.utils.model_loader import (
    robust_model_download,
    load_model_and_tokenizer,
    ModelDownloadError,
    _check_cached_model,
)


class TestModelLoader(unittest.TestCase):
    """Test cases for the model loader with retry mechanisms."""

    def test_retry_logic_with_network_failures(self):
        """Test that the loader retries on network failures."""
        # Mock the from_pretrained method to fail twice, then succeed
        mock_model = Mock()
        call_count = [0]
        
        def mock_from_pretrained(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Network error")
            return mock_model
        
        with patch('phaita.utils.model_loader.AutoModel') as mock_auto_model:
            mock_auto_model.from_pretrained = mock_from_pretrained
            
            # Should succeed after 2 retries
            result = robust_model_download(
                "test-model",
                model_type="auto",
                max_retries=3,
                timeout=10
            )
            
            self.assertEqual(result, mock_model)
            self.assertEqual(call_count[0], 3)  # Failed twice, succeeded on third
    
    def test_exponential_backoff_timing(self):
        """Test that exponential backoff increases wait time correctly."""
        start_times = []
        
        def mock_from_pretrained(*args, **kwargs):
            start_times.append(time.time())
            if len(start_times) < 3:
                raise ConnectionError("Network error")
            return Mock()
        
        with patch('phaita.utils.model_loader.AutoModel') as mock_auto_model:
            mock_auto_model.from_pretrained = mock_from_pretrained
            
            # Patch time.sleep to avoid actual waiting
            with patch('time.sleep') as mock_sleep:
                robust_model_download(
                    "test-model",
                    model_type="auto",
                    max_retries=3,
                    timeout=10
                )
                
                # Should have called sleep twice with exponential backoff
                self.assertEqual(mock_sleep.call_count, 2)
                # First backoff: 2^0 * 10 = 10s
                self.assertEqual(mock_sleep.call_args_list[0][0][0], 10)
                # Second backoff: 2^1 * 10 = 20s
                self.assertEqual(mock_sleep.call_args_list[1][0][0], 20)
    
    def test_offline_mode_uses_cached_models(self):
        """Test that offline mode only uses cached models."""
        mock_model = Mock()
        
        # Mock the cache check to return True
        with patch('phaita.utils.model_loader._check_cached_model', return_value=True):
            with patch('phaita.utils.model_loader.AutoModel') as mock_auto_model:
                mock_auto_model.from_pretrained = Mock(return_value=mock_model)
                
                result = robust_model_download(
                    "test-model",
                    model_type="auto",
                    allow_offline=True
                )
                
                self.assertEqual(result, mock_model)
                # Verify local_files_only was set to True
                call_kwargs = mock_auto_model.from_pretrained.call_args[1]
                self.assertTrue(call_kwargs.get('local_files_only'))
    
    def test_offline_mode_fails_without_cache(self):
        """Test that offline mode raises error when model not cached."""
        # Mock the cache check to return False
        with patch('phaita.utils.model_loader._check_cached_model', return_value=False):
            with self.assertRaises(ModelDownloadError) as cm:
                robust_model_download(
                    "test-model",
                    model_type="auto",
                    allow_offline=True
                )
            
            error_msg = str(cm.exception)
            self.assertIn("not found in cache", error_msg)
            self.assertIn("offline mode is enabled", error_msg)
            self.assertIn("huggingface-cli download", error_msg)
    
    def test_error_message_clarity_after_all_retries_fail(self):
        """Test that error messages are clear when all retries fail."""
        def mock_from_pretrained(*args, **kwargs):
            raise ConnectionError("Connection timeout")
        
        with patch('phaita.utils.model_loader.AutoModel') as mock_auto_model:
            mock_auto_model.from_pretrained = mock_from_pretrained
            
            # Patch time.sleep to avoid actual waiting
            with patch('time.sleep'):
                with self.assertRaises(ModelDownloadError) as cm:
                    robust_model_download(
                        "microsoft/deberta-v3-base",
                        model_type="auto",
                        max_retries=2,
                        timeout=10
                    )
                
                error_msg = str(cm.exception)
                # Check for helpful information in error message
                self.assertIn("Failed to download model", error_msg)
                self.assertIn("after 2 attempts", error_msg)
                self.assertIn("microsoft/deberta-v3-base", error_msg)
                self.assertIn("Troubleshooting", error_msg)
                self.assertIn("internet connection", error_msg)
                self.assertIn("huggingface-cli download", error_msg)
    
    def test_tokenizer_loading(self):
        """Test that tokenizer can be loaded separately."""
        mock_tokenizer = Mock()
        
        with patch('phaita.utils.model_loader.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_tokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
            
            result = robust_model_download(
                "test-model",
                model_type="tokenizer",
                max_retries=1
            )
            
            self.assertEqual(result, mock_tokenizer)
            self.assertTrue(mock_auto_tokenizer.from_pretrained.called)
    
    def test_causal_lm_loading(self):
        """Test that causal language models can be loaded."""
        mock_model = Mock()
        
        with patch('phaita.utils.model_loader.AutoModelForCausalLM') as mock_auto_model:
            mock_auto_model.from_pretrained = Mock(return_value=mock_model)
            
            result = robust_model_download(
                "test-model",
                model_type="causal_lm",
                max_retries=1
            )
            
            self.assertEqual(result, mock_model)
            self.assertTrue(mock_auto_model.from_pretrained.called)
    
    def test_load_model_and_tokenizer_together(self):
        """Test loading model and tokenizer together."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        with patch('phaita.utils.model_loader.AutoModel') as mock_auto_model:
            with patch('phaita.utils.model_loader.AutoTokenizer') as mock_auto_tokenizer:
                mock_auto_model.from_pretrained = Mock(return_value=mock_model)
                mock_auto_tokenizer.from_pretrained = Mock(return_value=mock_tokenizer)
                
                model, tokenizer = load_model_and_tokenizer(
                    "test-model",
                    model_type="auto",
                    max_retries=1
                )
                
                self.assertEqual(model, mock_model)
                self.assertEqual(tokenizer, mock_tokenizer)
    
    def test_resume_download_enabled(self):
        """Test that resume_download is enabled by default."""
        mock_model = Mock()
        
        with patch('phaita.utils.model_loader.AutoModel') as mock_auto_model:
            mock_auto_model.from_pretrained = Mock(return_value=mock_model)
            
            robust_model_download(
                "test-model",
                model_type="auto",
                max_retries=1
            )
            
            # Check that resume_download was passed
            call_kwargs = mock_auto_model.from_pretrained.call_args[1]
            self.assertTrue(call_kwargs.get('resume_download'))
    
    def test_invalid_model_type_raises_error(self):
        """Test that invalid model_type raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            robust_model_download(
                "test-model",
                model_type="invalid_type",
                max_retries=1
            )
        
        self.assertIn("model_type must be one of", str(cm.exception))
    
    def test_kwargs_passed_to_from_pretrained(self):
        """Test that additional kwargs are passed through."""
        mock_model = Mock()

        with patch('phaita.utils.model_loader.AutoModel') as mock_auto_model:
            mock_auto_model.from_pretrained = Mock(return_value=mock_model)

            robust_model_download(
                "test-model",
                model_type="auto",
                max_retries=1,
                device_map="auto",
                torch_dtype="float16"
            )

            # Check that custom kwargs were passed
            call_kwargs = mock_auto_model.from_pretrained.call_args[1]
            self.assertEqual(call_kwargs.get('device_map'), "auto")
            self.assertEqual(call_kwargs.get('torch_dtype'), "float16")

    def test_token_keyword_forwarded_to_transformers(self):
        """Token argument should be forwarded using the new keyword."""
        mock_model = Mock()

        with patch('phaita.utils.model_loader.AutoModel') as mock_auto_model:
            mock_auto_model.from_pretrained = Mock(return_value=mock_model)

            robust_model_download(
                "test-model",
                model_type="auto",
                max_retries=1,
                token="hf_secret"
            )

            call_kwargs = mock_auto_model.from_pretrained.call_args[1]
            self.assertEqual(call_kwargs.get('token'), "hf_secret")
            self.assertNotIn('use_auth_token', call_kwargs)

    def test_token_forwarded_for_model_and_tokenizer(self):
        """Token argument should be forwarded when loading both model and tokenizer."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        with patch('phaita.utils.model_loader.AutoModel') as mock_auto_model, \
                patch('phaita.utils.model_loader.AutoTokenizer') as mock_auto_tokenizer:
            mock_auto_model.from_pretrained = Mock(return_value=mock_model)
            mock_auto_tokenizer.from_pretrained = Mock(return_value=mock_tokenizer)

            model, tokenizer = load_model_and_tokenizer(
                "test-model",
                model_type="auto",
                max_retries=1,
                token="hf_secret"
            )

            self.assertEqual(model, mock_model)
            self.assertEqual(tokenizer, mock_tokenizer)

            model_kwargs = mock_auto_model.from_pretrained.call_args[1]
            tokenizer_kwargs = mock_auto_tokenizer.from_pretrained.call_args[1]

            self.assertEqual(model_kwargs.get('token'), "hf_secret")
            self.assertEqual(tokenizer_kwargs.get('token'), "hf_secret")
            self.assertNotIn('use_auth_token', model_kwargs)
            self.assertNotIn('use_auth_token', tokenizer_kwargs)


def run_tests():
    """Run all tests and report results."""
    print("🧪 Testing Model Loader with Retry Mechanisms")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelLoader)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report results
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun} tests passed")
    
    if result.wasSuccessful():
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
