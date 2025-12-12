"""
Speaker Diarization Module
==========================
Handles speaker identification and segmentation with pyannote.audio.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch
from pyannote.audio import Pipeline

from config import ProcessingConfig
from validation import ValidationError
from retry_utils import retry_on_exception

logger = logging.getLogger(__name__)


class DiarizationError(Exception):
    """Raised when diarization fails."""
    pass


class SpeakerDiarizer:
    """Speaker diarization using pyannote.audio."""
    
    def __init__(self, 
                 huggingface_token: str,
                 config: Optional[ProcessingConfig] = None):
        """
        Initialize speaker diarizer.
        
        Args:
            huggingface_token: HuggingFace authentication token
            config: Optional processing configuration
            
        Raises:
            ValidationError: If token is invalid
            DiarizationError: If pipeline initialization fails
        """
        if not huggingface_token:
            raise ValidationError("HuggingFace token is required")
        
        self.token = huggingface_token
        self.config = config or ProcessingConfig()
        self.pipeline = None
        self.device = self._setup_device()
        self._initialize_pipeline()
    
    def _setup_device(self) -> torch.device:
        """Setup compute device (GPU/CPU)."""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU (GPU not available)")
        
        return device
    
    @retry_on_exception(max_attempts=2, base_delay=5.0)
    def _initialize_pipeline(self) -> None:
        """Initialize pyannote pipeline with authentication."""
        try:
            logger.info("Loading pyannote speaker-diarization-3.1...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.token
            )
            self.pipeline.to(self.device)
            logger.info("✓ Diarization pipeline ready")
            
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            logger.info("\n💡 Troubleshooting:")
            logger.info("  1. Accept conditions: https://huggingface.co/pyannote/speaker-diarization-3.1")
            logger.info("  2. Accept conditions: https://huggingface.co/pyannote/segmentation-3.0")
            logger.info("  3. Get token: https://huggingface.co/settings/tokens")
            raise DiarizationError(f"Pipeline initialization failed: {e}") from e
    
    def diarize(self, 
                audio_path: Path,
                num_speakers: Optional[int] = None,
                min_segment_duration: Optional[float] = None) -> List[Dict]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Optional number of speakers (auto-detect if None)
            min_segment_duration: Minimum segment duration in seconds
            
        Returns:
            List of diarization segments with speaker labels
            
        Raises:
            DiarizationError: If diarization fails
        """
        if not audio_path.exists():
            raise ValidationError(f"Audio file not found: {audio_path}")
        
        num_speakers = num_speakers or self.config.num_speakers
        min_duration = min_segment_duration or self.config.min_segment_duration
        
        logger.info(f"Running diarization (num_speakers={num_speakers})...")
        
        try:
            # Run diarization
            diarization_result = self.pipeline(
                str(audio_path),
                num_speakers=num_speakers if num_speakers > 0 else None
            )
            
            # Extract segments
            segments = []
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                duration = turn.end - turn.start
                
                # Filter short segments
                if duration >= min_duration:
                    segments.append({
                        'start': turn.start,
                        'end': turn.end,
                        'duration': duration,
                        'speaker': speaker
                    })
            
            logger.info(f"✓ Found {len(segments)} segments from {num_speakers} speaker(s)")
            
            # Merge consecutive same-speaker segments
            merged_segments = self._merge_segments(segments)
            logger.info(f"✓ Merged to {len(merged_segments)} conversation turns")
            
            return merged_segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise DiarizationError(f"Failed to diarize audio: {e}") from e
    
    def _merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge consecutive segments from same speaker.
        
        Args:
            segments: List of diarization segments
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
        
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            gap = segment['start'] - current['end']
            same_speaker = segment['speaker'] == current['speaker']
            
            # Merge if same speaker and gap is small
            if same_speaker and gap <= self.config.merge_gap_threshold:
                current['end'] = segment['end']
                current['duration'] = current['end'] - current['start']
            else:
                merged.append(current)
                current = segment.copy()
        
        # Append last segment
        merged.append(current)
        
        return merged
    
    def extract_audio_segments(self,
                               audio_path: Path,
                               segments: List[Dict],
                               output_dir: Path) -> List[Dict]:
        """
        Extract individual audio segments to files.
        
        Args:
            audio_path: Path to source audio file
            segments: List of segment dictionaries
            output_dir: Directory to save extracted segments
            
        Returns:
            List of segments with 'path' field added
        """
        import torchaudio
        import soundfile as sf
        
        logger.info(f"Extracting {len(segments)} audio segments...")
        
        # Load audio once
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_segments = []
        
        for i, segment in enumerate(segments):
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            
            # Extract segment
            segment_audio = waveform[:, start_sample:end_sample]
            
            # Save to file
            segment_path = output_dir / f"segment_{i:04d}.wav"
            sf.write(
                str(segment_path),
                segment_audio.squeeze().numpy(),
                sample_rate
            )
            
            # Add path to segment info
            segment_copy = segment.copy()
            segment_copy['path'] = segment_path
            segment_copy['segment_id'] = i
            extracted_segments.append(segment_copy)
        
        logger.info(f"✓ Extracted {len(extracted_segments)} segments to {output_dir}")
        return extracted_segments
    
    def get_speaker_statistics(self, segments: List[Dict]) -> Dict:
        """
        Calculate speaker statistics from segments.
        
        Args:
            segments: List of diarization segments
            
        Returns:
            Dictionary with speaker statistics
        """
        from collections import defaultdict
        
        stats = defaultdict(lambda: {'count': 0, 'total_duration': 0.0})
        
        for segment in segments:
            speaker = segment['speaker']
            stats[speaker]['count'] += 1
            stats[speaker]['total_duration'] += segment['duration']
        
        # Calculate speaking percentages
        total_duration = sum(s['total_duration'] for s in stats.values())
        
        for speaker, data in stats.items():
            data['percentage'] = (data['total_duration'] / total_duration * 100) if total_duration > 0 else 0
            data['avg_segment_duration'] = data['total_duration'] / data['count'] if data['count'] > 0 else 0
        
        return dict(stats)


if __name__ == "__main__":
    # Test diarization
    import os
    from dotenv import load_dotenv
    
    load_dotenv(".env.local")
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("❌ HUGGINGFACE_TOKEN not found in environment")
        exit(1)
    
    audio_file = Path(r"D:\Skipper\Call-optimaization\temp\test_cleaned.wav")
    if not audio_file.exists():
        print(f"❌ Test file not found: {audio_file}")
        exit(1)
    
    try:
        diarizer = SpeakerDiarizer(huggingface_token=token)
        segments = diarizer.diarize(audio_file, num_speakers=2)
        
        print(f"\n✅ Diarization complete: {len(segments)} segments")
        
        # Print statistics
        stats = diarizer.get_speaker_statistics(segments)
        print("\nSpeaker Statistics:")
        for speaker, data in stats.items():
            print(f"  {speaker}: {data['count']} segments, "
                  f"{data['total_duration']:.1f}s ({data['percentage']:.1f}%)")
    
    except Exception as e:
        print(f"❌ Error: {e}")
