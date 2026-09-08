"""
Transcription Module
===================
Handles speech-to-text using Sarvam AI with proper error handling and retries.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from sarvamai import SarvamAI

from .config import TranslationConfig
from .validation import ValidationError
from .retry_utils import retry_on_exception

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


class SarvamTranscriber:
    """Speech-to-text transcription using Sarvam AI."""
    
    def __init__(self, 
                 api_key: str,
                 config: Optional[TranslationConfig] = None):
        """
        Initialize transcriber.
        
        Args:
            api_key: Sarvam AI API key
            config: Optional translation configuration
            
        Raises:
            ValidationError: If API key is invalid
        """
        if not api_key:
            raise ValidationError("Sarvam API key is required")
        
        self.api_key = api_key
        self.config = config or TranslationConfig()
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Sarvam API client."""
        try:
            self.client = SarvamAI(api_subscription_key=self.api_key)
            logger.info("✓ Sarvam AI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Sarvam client: {e}")
            raise TranscriptionError(f"Client initialization failed: {e}") from e
    
    @retry_on_exception(
        exceptions=(ConnectionError, TimeoutError),
        max_attempts=3,
        base_delay=2.0,
        max_delay=30.0
    )
    def transcribe_file(self, 
                       audio_path: Path,
                       language_code: Optional[str] = None,
                       model: str = "saarika:v2.5") -> str:
        """
        Transcribe single audio file.
        
        Args:
            audio_path: Path to audio file
            language_code: Language code (default: from config)
            model: Sarvam model to use
            
        Returns:
            Transcribed text
            
        Raises:
            TranscriptionError: If transcription fails
        """
        if not audio_path.exists():
            raise ValidationError(f"Audio file not found: {audio_path}")
        
        language_code = language_code or self.config.source_language
        
        try:
            with open(audio_path, 'rb') as audio_file:
                response = self.client.speech_to_text.transcribe(
                    file=audio_file,
                    model=model,
                    language_code=language_code
                )
            
            # Extract text from response
            text = self._extract_text(response)
            
            if not text or not text.strip():
                logger.warning(f"Empty transcription for {audio_path.name}")
                return ""
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path.name}: {e}")
            raise TranscriptionError(f"Failed to transcribe {audio_path.name}: {e}") from e
    
    def _extract_text(self, response) -> str:
        """
        Extract text from API response.
        
        Args:
            response: API response object
            
        Returns:
            Extracted text
        """
        # Handle different response formats
        if isinstance(response, dict):
            return response.get("text") or response.get("transcript") or ""
        
        # Handle object response
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "transcript"):
            return response.transcript
        
        logger.warning(f"Unknown response format: {type(response)}")
        return str(response)
    
    def transcribe_segments(self, 
                           segments: List[Dict],
                           show_progress: bool = True) -> List[Dict]:
        """
        Transcribe multiple audio segments.
        
        Args:
            segments: List of segment dictionaries with 'path' field
            show_progress: Show progress bar
            
        Returns:
            Segments with 'text' field added
        """
        logger.info(f"Transcribing {len(segments)} segments...")
        
        transcribed = []
        failed_count = 0
        
        iterator = segments
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(segments, desc="Transcribing")
            except ImportError:
                pass
        
        for i, segment in enumerate(iterator):
            if 'path' not in segment:
                logger.warning(f"Segment {i} missing 'path' field, skipping")
                failed_count += 1
                continue
            
            try:
                text = self.transcribe_file(segment['path'])
                
                if text:
                    segment_copy = segment.copy()
                    segment_copy['text'] = text
                    segment_copy['language'] = self.config.source_language
                    transcribed.append(segment_copy)
                else:
                    logger.warning(f"Empty transcription for segment {i}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to transcribe segment {i}: {e}")
                failed_count += 1
                continue
        
        success_rate = (len(transcribed) / len(segments) * 100) if segments else 0
        logger.info(f"✓ Transcribed {len(transcribed)}/{len(segments)} segments ({success_rate:.1f}% success)")
        
        if failed_count > 0:
            logger.warning(f"⚠ {failed_count} segments failed transcription")
        
        return transcribed
    
    def batch_transcribe(self,
                        audio_files: List[Path],
                        language_code: Optional[str] = None) -> Dict[Path, str]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            language_code: Optional language code
            
        Returns:
            Dictionary mapping file paths to transcriptions
        """
        results = {}
        
        for audio_file in audio_files:
            try:
                text = self.transcribe_file(audio_file, language_code)
                results[audio_file] = text
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file.name}: {e}")
                results[audio_file] = None
        
        return results
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of language codes
        """
        # Sarvam AI supported languages (as of 2024)
        return [
            "hi-IN",  # Hindi
            "en-IN",  # English (Indian)
            "bn-IN",  # Bengali
            "ta-IN",  # Tamil
            "te-IN",  # Telugu
            "mr-IN",  # Marathi
            "gu-IN",  # Gujarati
            "kn-IN",  # Kannada
            "ml-IN",  # Malayalam
            "pa-IN",  # Punjabi
        ]


if __name__ == "__main__":
    # Test transcription
    import os
    from dotenv import load_dotenv
    
    load_dotenv(".env.local")
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        print("❌ SARVAM_API_KEY not found in environment")
        exit(1)
    
    # Test with a sample file
    test_file = Path(r"D:\Skipper\Call-optimaization\temp\segment_0000.wav")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        exit(1)
    
    try:
        transcriber = SarvamTranscriber(api_key=api_key)
        text = transcriber.transcribe_file(test_file)
        print(f"\n✅ Transcription: {text}")
    except Exception as e:
        print(f"❌ Error: {e}")
