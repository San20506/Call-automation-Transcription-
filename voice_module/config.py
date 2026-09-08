"""
Configuration Management
========================
Centralized, validated configuration with environment variable support.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")


@dataclass
class APIConfig:
    """API credentials and endpoints."""
    
    sarvam_api_key: str = field(default_factory=lambda: os.getenv("SARVAM_API_KEY", ""))
    huggingface_token: str = field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    
    def validate(self) -> None:
        """Validate that all required API keys are present."""
        if not self.sarvam_api_key:
            raise ValueError(
                "SARVAM_API_KEY not found in environment. "
                "Please set it in .env.local or environment variables."
            )
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Get one from: https://aistudio.google.com/app/apikey"
            )
        if not self.huggingface_token:
            raise ValueError(
                "HUGGINGFACE_TOKEN not found in environment. "
                "Get one from: https://huggingface.co/settings/tokens"
            )


@dataclass
class PathConfig:
    """File system paths."""
    
    input_dir: Path = field(default_factory=lambda: Path("input"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    temp_dir: Path = field(default_factory=lambda: Path("temp"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    def __post_init__(self):
        """Ensure all paths are Path objects."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.temp_dir = Path(self.temp_dir)
        self.log_dir = Path(self.log_dir)
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        for path in [self.output_dir, self.temp_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> None:
        """Validate path configuration."""
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")


@dataclass
class ProcessingConfig:
    """Audio processing parameters."""
    
    # Audio constraints
    max_file_size_mb: int = 500
    max_duration_seconds: int = 3600  # 1 hour
    min_duration_seconds: float = 0.5
    
    # Supported formats
    supported_formats: tuple = ('.mp3', '.wav', '.m4a', '.flac')
    
    # Processing parameters
    use_gpu: bool = field(default_factory=lambda: _check_gpu())
    min_segment_duration: float = 0.5
    merge_gap_threshold: float = 1.0
    num_speakers: int = 2
    
    # Performance
    batch_size: int = 10
    num_workers: int = 4
    timeout_seconds: int = 300
    
    def validate(self) -> None:
        """Validate processing parameters."""
        if self.min_segment_duration < 0:
            raise ValueError("min_segment_duration must be positive")
        if self.merge_gap_threshold < 0:
            raise ValueError("merge_gap_threshold must be positive")
        if self.num_speakers < 1:
            raise ValueError("num_speakers must be at least 1")
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")


@dataclass
class TranslationConfig:
    """Translation and analysis settings."""
    
    source_language: str = "hi-IN"
    target_language: str = "en-IN"
    model: str = "gemini-1.5-flash"
    max_tokens: int = 8000
    temperature: float = 0.3
    chunk_size: int = 4500
    enable_caching: bool = True
    
    def validate(self) -> None:
        """Validate translation parameters."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens < 100:
            raise ValueError("max_tokens must be at least 100")


@dataclass
class RetryConfig:
    """Retry and resilience settings."""
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    
    def validate(self) -> None:
        """Validate retry parameters."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")


@dataclass
class Config:
    """Master configuration object."""
    
    api: APIConfig = field(default_factory=APIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    def validate_all(self) -> None:
        """Validate entire configuration."""
        self.api.validate()
        self.paths.validate()
        self.processing.validate()
        self.translation.validate()
        self.retry.validate()
    
    def setup(self) -> None:
        """Setup configuration (create dirs, validate, etc.)."""
        self.paths.create_directories()
        self.validate_all()


def _check_gpu() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Optional path to config file (JSON/YAML)
        
    Returns:
        Configured Config object
    """
    cfg = Config()
    
    # TODO: Add support for loading from JSON/YAML if needed
    if config_path and config_path.exists():
        pass  # Future: load from file
    
    cfg.setup()
    return cfg


if __name__ == "__main__":
    # Test configuration
    try:
        cfg = load_config()
        print("✅ Configuration loaded successfully")
        print(f"GPU Available: {cfg.processing.use_gpu}")
        print(f"Output Dir: {cfg.paths.output_dir}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
