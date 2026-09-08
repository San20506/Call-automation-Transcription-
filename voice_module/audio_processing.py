"""
Audio Processing Module
=======================
Handles noise reduction and audio preprocessing with proper error handling.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from scipy import signal
import librosa
import noisereduce as nr
import soundfile as sf

from .validation import AudioValidator, ValidationError
from .retry_utils import retry_file_operation

logger = logging.getLogger(__name__)


class NoiseReductionError(Exception):
    """Raised when noise reduction fails."""
    pass


class AudioCleaner:
    """Multi-stage audio noise reduction and enhancement."""
    
    def __init__(self, validator: Optional[AudioValidator] = None):
        """
        Initialize audio cleaner.
        
        Args:
            validator: Optional AudioValidator instance
        """
        self.validator = validator or AudioValidator()
    
    @retry_file_operation
    def clean_audio(self, 
                   input_path: Path, 
                   output_path: Path,
                   stationary_prop: float = 0.90,
                   non_stationary_prop: float = 0.70) -> Path:
        """
        Clean noisy audio with multi-stage processing.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save cleaned audio
            stationary_prop: Proportion of stationary noise to reduce
            non_stationary_prop: Proportion of non-stationary noise to reduce
            
        Returns:
            Path to cleaned audio file
            
        Raises:
            NoiseReductionError: If cleaning fails
            ValidationError: If input file is invalid
        """
        # Validate input
        self.validator.validate_and_raise(str(input_path))
        
        logger.info(f"Cleaning audio: {input_path.name}")
        
        try:
            # Load audio
            y, sr = librosa.load(str(input_path), sr=None)
            duration = len(y) / sr
            logger.info(f"  Duration: {duration:.1f}s | Sample rate: {sr}Hz")
            
            # Stage 1: Remove stationary noise (AC hum, background noise)
            logger.info("  [1/4] Removing stationary noise...")
            y = nr.reduce_noise(
                y=y,
                sr=sr,
                stationary=True,
                prop_decrease=stationary_prop,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50
            )
            
            # Stage 2: Remove non-stationary noise (traffic, voices)
            logger.info("  [2/4] Removing non-stationary noise...")
            y = nr.reduce_noise(
                y=y,
                sr=sr,
                stationary=False,
                prop_decrease=non_stationary_prop,
                n_std_thresh_stationary=1.2,
                time_constant_s=2.0
            )
            
            # Stage 3: Filter low-frequency rumble
            logger.info("  [3/4] Filtering frequencies...")
            y = self._apply_filters(y, sr)
            
            # Stage 4: Normalize audio
            logger.info("  [4/4] Normalizing audio...")
            y = self._normalize_audio(y)
            
            # Save cleaned audio
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), y, sr, subtype='PCM_16')
            
            logger.info(f"  ✓ Cleaned audio saved: {output_path.name}")
            return output_path
            
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            raise NoiseReductionError(f"Failed to clean audio: {e}") from e
    
    def _apply_filters(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply highpass and lowpass filters.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Filtered audio
        """
        # Highpass filter (remove low-frequency rumble)
        highpass_freq = min(80, sample_rate * 0.01)
        sos_high = signal.butter(4, highpass_freq, 'hp', fs=sample_rate, output='sos')
        audio = signal.sosfilt(sos_high, audio)
        
        # Lowpass filter (remove high-frequency hiss)
        nyquist = sample_rate / 2
        lowpass_freq = min(nyquist * 0.95, 8000)
        sos_low = signal.butter(4, lowpass_freq, 'lp', fs=sample_rate, output='sos')
        audio = signal.sosfilt(sos_low, audio)
        
        return audio
    
    def _normalize_audio(self, audio: np.ndarray, target_peak: float = 0.707) -> np.ndarray:
        """
        Normalize and compress audio.
        
        Args:
            audio: Audio signal
            target_peak: Target peak amplitude (default: -3dB)
            
        Returns:
            Normalized audio
        """
        # Normalize to target peak
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * target_peak
        
        # Simple compression (reduce dynamic range)
        threshold = 0.3
        ratio = 3.0
        compressed = np.where(
            np.abs(audio) > threshold,
            np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
            audio
        )
        
        # Apply makeup gain and clip
        audio = np.clip(compressed * 1.5, -1.0, 1.0)
        
        return audio
    
    def get_audio_info(self, file_path: Path) -> dict:
        """
        Get audio file information without loading entire file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio metadata
        """
        try:
            import torchaudio
            info = torchaudio.info(str(file_path))
            
            return {
                'duration': info.num_frames / info.sample_rate,
                'sample_rate': info.sample_rate,
                'num_channels': info.num_channels,
                'num_frames': info.num_frames,
                'encoding': info.encoding if hasattr(info, 'encoding') else 'unknown'
            }
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return {}


class BatchAudioCleaner:
    """Process multiple audio files in batch."""
    
    def __init__(self, cleaner: Optional[AudioCleaner] = None):
        """
        Initialize batch cleaner.
        
        Args:
            cleaner: Optional AudioCleaner instance
        """
        self.cleaner = cleaner or AudioCleaner()
    
    def clean_directory(self,
                       input_dir: Path,
                       output_dir: Path,
                       pattern: str = "*.mp3") -> Tuple[list, list]:
        """
        Clean all audio files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            pattern: File pattern to match (default: "*.mp3")
            
        Returns:
            Tuple of (successful_files, failed_files)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all matching files
        audio_files = list(input_dir.glob(pattern))
        if not audio_files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return [], []
        
        logger.info(f"Found {len(audio_files)} files to process")
        
        successful = []
        failed = []
        
        for i, input_file in enumerate(audio_files, 1):
            logger.info(f"\n[{i}/{len(audio_files)}] Processing {input_file.name}")
            
            try:
                output_file = output_dir / f"cleaned_{input_file.stem}.wav"
                self.cleaner.clean_audio(input_file, output_file)
                successful.append(input_file)
            except Exception as e:
                logger.error(f"  ❌ Failed: {e}")
                failed.append((input_file, str(e)))
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch processing complete:")
        logger.info(f"  Successful: {len(successful)}/{len(audio_files)}")
        logger.info(f"  Failed: {len(failed)}/{len(audio_files)}")
        logger.info(f"{'='*60}")
        
        return successful, failed


if __name__ == "__main__":
    # Test audio cleaning
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    cleaner = AudioCleaner()
    
    # Test single file
    input_file = Path(r"D:\Skipper\Call-optimaization\input\noisy_Voice.mp3")
    output_file = Path(r"D:\Skipper\Call-optimaization\temp\test_cleaned.wav")
    
    if input_file.exists():
        try:
            result = cleaner.clean_audio(input_file, output_file)
            print(f"✅ Success: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        print(f"❌ Test file not found: {input_file}")
