"""
Input Validation Module
=======================
Comprehensive validation for audio files and processing parameters.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import mimetypes
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class AudioValidator:
    """Validates audio files before processing."""
    
    def __init__(self, 
                 max_size_mb: int = 500,
                 max_duration_seconds: int = 3600,
                 min_duration_seconds: float = 0.5,
                 supported_formats: tuple = ('.mp3', '.wav', '.m4a', '.flac')):
        """
        Initialize validator with constraints.
        
        Args:
            max_size_mb: Maximum file size in megabytes
            max_duration_seconds: Maximum audio duration in seconds
            min_duration_seconds: Minimum audio duration in seconds
            supported_formats: Tuple of supported file extensions
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_duration = max_duration_seconds
        self.min_duration = min_duration_seconds
        self.supported_formats = supported_formats
    
    def validate_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate audio file exists and meets requirements.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(file_path)
        
        # Check existence
        if not path.exists():
            return False, f"File does not exist: {file_path}"
        
        if not path.is_file():
            return False, f"Path is not a file: {file_path}"
        
        # Check file size
        size = path.stat().st_size
        if size == 0:
            return False, "File is empty"
        
        if size > self.max_size_bytes:
            size_mb = size / (1024 * 1024)
            max_mb = self.max_size_bytes / (1024 * 1024)
            return False, f"File too large: {size_mb:.1f}MB (max: {max_mb}MB)"
        
        # Check format
        ext = path.suffix.lower()
        if ext not in self.supported_formats:
            return False, f"Unsupported format: {ext}. Supported: {self.supported_formats}"
        
        # Verify it's actually an audio file (MIME type check)
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type and not mime_type.startswith('audio'):
            return False, f"File is not an audio file (MIME: {mime_type})"
        
        # Check if file is corrupted (try to read headers)
        try:
            is_valid, error = self._validate_audio_headers(path)
            if not is_valid:
                return False, error
        except Exception as e:
            return False, f"Failed to read audio file: {str(e)}"
        
        return True, None
    
    def _validate_audio_headers(self, path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate audio file can be read and get duration.
        
        Args:
            path: Path to audio file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            import torchaudio
            
            # Try to load audio info without loading entire file
            info = torchaudio.info(str(path))
            duration = info.num_frames / info.sample_rate
            
            # Check duration constraints
            if duration < self.min_duration:
                return False, f"Audio too short: {duration:.1f}s (min: {self.min_duration}s)"
            
            if duration > self.max_duration:
                return False, f"Audio too long: {duration:.1f}s (max: {self.max_duration}s)"
            
            # Check sample rate is reasonable
            if info.sample_rate < 8000 or info.sample_rate > 192000:
                return False, f"Invalid sample rate: {info.sample_rate}Hz"
            
            logger.info(f"Audio validated: {duration:.1f}s, {info.sample_rate}Hz, {info.num_channels}ch")
            return True, None
            
        except Exception as e:
            return False, f"Corrupted or invalid audio file: {str(e)}"
    
    def validate_and_raise(self, file_path: str) -> None:
        """
        Validate file and raise ValidationError if invalid.
        
        Args:
            file_path: Path to audio file
            
        Raises:
            ValidationError: If validation fails
        """
        is_valid, error = self.validate_file(file_path)
        if not is_valid:
            raise ValidationError(error)


class ConfigValidator:
    """Validates configuration parameters."""
    
    @staticmethod
    def validate_api_key(key: Optional[str], name: str) -> None:
        """
        Validate API key format.
        
        Args:
            key: API key string
            name: Name of the API key (for error messages)
            
        Raises:
            ValidationError: If key is invalid
        """
        if not key:
            raise ValidationError(f"{name} is required but not provided")
        
        if not isinstance(key, str):
            raise ValidationError(f"{name} must be a string")
        
        if len(key) < 10:
            raise ValidationError(f"{name} appears to be too short (possibly invalid)")
        
        # Check for common mistakes
        if key.startswith("YOUR_") or key == "PLACEHOLDER":
            raise ValidationError(f"{name} is a placeholder, not a real key")
    
    @staticmethod
    def validate_directory(path: Path, create: bool = False) -> None:
        """
        Validate directory exists or can be created.
        
        Args:
            path: Directory path
            create: If True, create directory if it doesn't exist
            
        Raises:
            ValidationError: If directory is invalid
        """
        if path.exists():
            if not path.is_dir():
                raise ValidationError(f"Path exists but is not a directory: {path}")
            if not os.access(path, os.W_OK):
                raise ValidationError(f"Directory is not writable: {path}")
        else:
            if create:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValidationError(f"Cannot create directory {path}: {e}")
            else:
                raise ValidationError(f"Directory does not exist: {path}")
    
    @staticmethod
    def validate_language_code(code: str) -> None:
        """
        Validate language code format.
        
        Args:
            code: Language code (e.g., 'hi-IN', 'en-US')
            
        Raises:
            ValidationError: If code is invalid
        """
        if not code:
            raise ValidationError("Language code is required")
        
        # Basic format check (language-COUNTRY)
        if '-' not in code or len(code.split('-')) != 2:
            raise ValidationError(f"Invalid language code format: {code}. Expected: 'xx-XX'")
        
        lang, country = code.split('-')
        if len(lang) != 2 or len(country) != 2:
            raise ValidationError(f"Invalid language code: {code}")
        
        if not lang.islower() or not country.isupper():
            raise ValidationError(f"Language code format error: {code}. Use lowercase-UPPERCASE")


def validate_processing_ready() -> Tuple[bool, list]:
    """
    Check if system is ready for processing.
    
    Returns:
        Tuple of (is_ready, list_of_issues)
    """
    issues = []
    
    # Check PyTorch
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("GPU not available (CPU will be used, slower)")
    except ImportError:
        issues.append("PyTorch not installed")
    
    # Check audio libraries
    required_modules = ['librosa', 'soundfile', 'noisereduce', 'scipy']
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            issues.append(f"Required module not installed: {module}")
    
    # Check API libraries
    api_modules = ['pyannote.audio', 'sarvamai', 'google.generativeai']
    for module in api_modules:
        try:
            __import__(module)
        except ImportError:
            issues.append(f"API module not installed: {module}")
    
    return len(issues) == 0, issues


if __name__ == "__main__":
    # Test validation
    validator = AudioValidator()
    
    test_file = r"D:\Skipper\Call-optimaization\input\noisy_Voice.mp3"
    is_valid, error = validator.validate_file(test_file)
    
    if is_valid:
        print(f"✅ File is valid: {test_file}")
    else:
        print(f"❌ Validation failed: {error}")
    
    # Check system readiness
    ready, issues = validate_processing_ready()
    if ready:
        print("✅ System ready for processing")
    else:
        print("❌ System not ready:")
        for issue in issues:
            print(f"  - {issue}")
