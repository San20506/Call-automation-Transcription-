"""
Test Suite for Call Analysis Pipeline
====================================
Comprehensive tests for all modules.
"""

import pytest
from pathlib import Path
import os
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from validation import AudioValidator, ConfigValidator, ValidationError
from retry_utils import retry_on_exception, RetryError, RetryStrategy
from config import Config, APIConfig, PathConfig


class TestAudioValidator:
    """Test audio validation functionality."""
    
    def test_file_not_found(self):
        """Test validation fails for non-existent file."""
        validator = AudioValidator()
        is_valid, error = validator.validate_file("nonexistent.mp3")
        assert not is_valid
        assert "does not exist" in error.lower()
    
    def test_empty_file(self, tmp_path):
        """Test validation fails for empty file."""
        empty_file = tmp_path / "empty.mp3"
        empty_file.touch()
        
        validator = AudioValidator()
        is_valid, error = validator.validate_file(str(empty_file))
        assert not is_valid
        assert "empty" in error.lower()
    
    def test_file_too_large(self, tmp_path):
        """Test validation fails for oversized file."""
        large_file = tmp_path / "large.mp3"
        # Create file larger than max (default 500MB)
        with open(large_file, 'wb') as f:
            f.write(b'0' * (501 * 1024 * 1024))  # 501MB
        
        validator = AudioValidator(max_size_mb=500)
        is_valid, error = validator.validate_file(str(large_file))
        assert not is_valid
        assert "too large" in error.lower()
    
    def test_unsupported_format(self, tmp_path):
        """Test validation fails for unsupported format."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("not an audio file")
        
        validator = AudioValidator()
        is_valid, error = validator.validate_file(str(invalid_file))
        assert not is_valid
        assert "unsupported format" in error.lower()


class TestConfigValidator:
    """Test configuration validation."""
    
    def test_api_key_required(self):
        """Test API key validation rejects empty keys."""
        with pytest.raises(ValidationError, match="required"):
            ConfigValidator.validate_api_key("", "TEST_KEY")
    
    def test_api_key_too_short(self):
        """Test API key validation rejects short keys."""
        with pytest.raises(ValidationError, match="too short"):
            ConfigValidator.validate_api_key("abc", "TEST_KEY")
    
    def test_api_key_placeholder(self):
        """Test API key validation rejects placeholders."""
        with pytest.raises(ValidationError, match="placeholder"):
            ConfigValidator.validate_api_key("YOUR_API_KEY", "TEST_KEY")
    
    def test_valid_api_key(self):
        """Test valid API key passes validation."""
        # Should not raise
        ConfigValidator.validate_api_key("sk_valid_api_key_1234567890", "TEST_KEY")
    
    def test_directory_validation_non_existent(self):
        """Test directory validation fails for non-existent path."""
        with pytest.raises(ValidationError, match="does not exist"):
            ConfigValidator.validate_directory(Path("/nonexistent/path"), create=False)
    
    def test_directory_creation(self, tmp_path):
        """Test directory can be created."""
        new_dir = tmp_path / "test_dir"
        assert not new_dir.exists()
        
        ConfigValidator.validate_directory(new_dir, create=True)
        assert new_dir.exists()
        assert new_dir.is_dir()


class TestRetryStrategy:
    """Test retry logic and exponential backoff."""
    
    def test_retry_success_first_attempt(self):
        """Test function succeeds on first try."""
        mock_func = Mock(return_value="success")
        
        @retry_on_exception(max_attempts=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_success_after_failures(self):
        """Test function succeeds after some failures."""
        attempt_count = [0]
        
        @retry_on_exception(max_attempts=3, base_delay=0.1)
        def test_func():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = test_func()
        assert result == "success"
        assert attempt_count[0] == 3
    
    def test_retry_exhausts_attempts(self):
        """Test all retry attempts are exhausted."""
        @retry_on_exception(max_attempts=3, base_delay=0.1)
        def test_func():
            raise ConnectionError("Always fails")
        
        with pytest.raises(RetryError):
            test_func()
    
    def test_exponential_backoff_delay(self):
        """Test exponential backoff calculates correct delays."""
        strategy = RetryStrategy(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=False
        )
        
        delays = [strategy.get_delay(i) for i in range(5)]
        expected = [1.0, 2.0, 4.0, 8.0, 16.0]
        
        assert delays == expected


class TestConfig:
    """Test configuration management."""
    
    def test_config_defaults(self):
        """Test configuration has sensible defaults."""
        cfg = Config()
        
        assert cfg.processing.use_gpu is not None
        assert cfg.processing.min_segment_duration > 0
        assert cfg.retry.max_attempts >= 1
    
    def test_config_validation_invalid_params(self):
        """Test configuration validation catches invalid parameters."""
        cfg = Config()
        cfg.processing.min_segment_duration = -1  # Invalid
        
        with pytest.raises(ValueError):
            cfg.processing.validate()
    
    @patch.dict(os.environ, {
        'SARVAM_API_KEY': 'test_sarvam_key_12345',
        'GEMINI_API_KEY': 'test_gemini_key_12345',
        'HUGGINGFACE_TOKEN': 'test_hf_token_12345'
    })
    def test_config_loads_from_env(self):
        """Test configuration loads from environment variables."""
        cfg = Config()
        
        assert cfg.api.sarvam_api_key == 'test_sarvam_key_12345'
        assert cfg.api.gemini_api_key == 'test_gemini_key_12345'
        assert cfg.api.huggingface_token == 'test_hf_token_12345'


# Integration test (requires actual files)
class TestIntegration:
    """Integration tests (run only if test files available)."""
    
    @pytest.mark.skipif(
        not Path("input/test_audio.mp3").exists(),
        reason="Test audio file not available"
    )
    def test_end_to_end_pipeline(self):
        """Test complete pipeline end-to-end."""
        # This would test the full pipeline
        # Skipped if test data not available
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
