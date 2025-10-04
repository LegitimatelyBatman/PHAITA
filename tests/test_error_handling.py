"""
Test suite for improved error handling in _load_llm methods.
Tests specific exception catching and logging behavior.
"""

import logging
from unittest.mock import Mock, patch, call
import pytest

from phaita.utils.model_loader import ModelDownloadError


def test_generator_catches_model_download_error():
    """Test that ComplaintGenerator catches ModelDownloadError specifically."""
    from phaita.models.generator import ComplaintGenerator
    
    with patch('phaita.models.generator.load_model_and_tokenizer') as mock_loader:
        mock_loader.side_effect = ModelDownloadError("Network error during download")
        
        with pytest.raises(RuntimeError) as exc_info:
            generator = ComplaintGenerator(use_pretrained=True)
        
        assert "Failed to load model" in str(exc_info.value)
        assert "Network error during download" in str(exc_info.value)


def test_generator_catches_file_not_found_error():
    """Test that ComplaintGenerator catches FileNotFoundError specifically."""
    from phaita.models.generator import ComplaintGenerator
    
    with patch('phaita.models.generator.load_model_and_tokenizer') as mock_loader:
        mock_loader.side_effect = FileNotFoundError("Model file not found")
        
        with pytest.raises(RuntimeError) as exc_info:
            generator = ComplaintGenerator(use_pretrained=True)
        
        assert "Failed to load model" in str(exc_info.value)
        assert "Model files not found" in str(exc_info.value)
        assert "may not have been downloaded" in str(exc_info.value)


def test_generator_catches_oserror():
    """Test that ComplaintGenerator catches OSError specifically."""
    from phaita.models.generator import ComplaintGenerator
    
    with patch('phaita.models.generator.load_model_and_tokenizer') as mock_loader:
        mock_loader.side_effect = OSError("Disk I/O error")
        
        with pytest.raises(RuntimeError) as exc_info:
            generator = ComplaintGenerator(use_pretrained=True)
        
        assert "Failed to load model" in str(exc_info.value)
        assert "OSError" in str(exc_info.value)


def test_generator_catches_value_error():
    """Test that ComplaintGenerator catches ValueError specifically."""
    from phaita.models.generator import ComplaintGenerator
    
    with patch('phaita.models.generator.load_model_and_tokenizer') as mock_loader:
        mock_loader.side_effect = ValueError("Invalid configuration")
        
        with pytest.raises(RuntimeError) as exc_info:
            generator = ComplaintGenerator(use_pretrained=True)
        
        assert "Failed to load model" in str(exc_info.value)
        assert "ValueError" in str(exc_info.value)


def test_generator_logs_errors():
    """Test that ComplaintGenerator logs errors before raising."""
    from phaita.models.generator import ComplaintGenerator
    
    with patch('phaita.models.generator.load_model_and_tokenizer') as mock_loader:
        with patch('phaita.models.generator.logger') as mock_logger:
            mock_loader.side_effect = OSError("Disk I/O error")
            
            try:
                generator = ComplaintGenerator(use_pretrained=True)
            except RuntimeError:
                pass  # Expected
            
            # Verify that logger.error was called
            assert mock_logger.error.called
            error_call = mock_logger.error.call_args[0][0]
            assert "Model loading failed" in error_call
            assert "OSError" in error_call


def test_question_generator_catches_model_download_error():
    """Test that QuestionGenerator catches ModelDownloadError specifically."""
    from phaita.models.question_generator import QuestionGenerator
    
    with patch('phaita.models.question_generator.load_model_and_tokenizer') as mock_loader:
        mock_loader.side_effect = ModelDownloadError("Network error during download")
        
        with pytest.raises(RuntimeError) as exc_info:
            generator = QuestionGenerator(use_pretrained=True)
        
        assert "Failed to load model" in str(exc_info.value)
        assert "Network error during download" in str(exc_info.value)


def test_question_generator_catches_file_not_found_error():
    """Test that QuestionGenerator catches FileNotFoundError specifically."""
    from phaita.models.question_generator import QuestionGenerator
    
    with patch('phaita.models.question_generator.load_model_and_tokenizer') as mock_loader:
        mock_loader.side_effect = FileNotFoundError("Model file not found")
        
        with pytest.raises(RuntimeError) as exc_info:
            generator = QuestionGenerator(use_pretrained=True)
        
        assert "Failed to load model" in str(exc_info.value)
        assert "Model files not found" in str(exc_info.value)


def test_question_generator_logs_errors():
    """Test that QuestionGenerator logs errors before raising."""
    from phaita.models.question_generator import QuestionGenerator
    
    with patch('phaita.models.question_generator.load_model_and_tokenizer') as mock_loader:
        with patch('phaita.models.question_generator.logger') as mock_logger:
            mock_loader.side_effect = OSError("Disk I/O error")
            
            try:
                generator = QuestionGenerator(use_pretrained=True)
            except RuntimeError:
                pass  # Expected
            
            # Verify that logger.error was called
            assert mock_logger.error.called
            error_call = mock_logger.error.call_args[0][0]
            assert "Model loading failed" in error_call
            assert "OSError" in error_call


def test_error_messages_include_exception_type():
    """Test that error messages include the specific exception type name."""
    from phaita.models.generator import ComplaintGenerator
    
    with patch('phaita.models.generator.load_model_and_tokenizer') as mock_loader:
        mock_loader.side_effect = ValueError("Invalid model configuration")
        
        with pytest.raises(RuntimeError) as exc_info:
            generator = ComplaintGenerator(use_pretrained=True)
        
        error_msg = str(exc_info.value)
        assert "ValueError" in error_msg
        assert "Invalid model configuration" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
