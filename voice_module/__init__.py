"""
Voice Transcription Module
==========================
End-to-end call analysis pipeline:
audio cleaning → diarization → transcription → translation → analysis.

Usage:
    from voice_module import CallAnalysisPipeline

    pipeline = CallAnalysisPipeline()
    results = pipeline.process(Path("call.mp3"))
"""

from .pipeline import CallAnalysisPipeline
from .transcription import SarvamTranscriber
from .audio_processing import AudioCleaner
from .diarization import SpeakerDiarizer
from .translation import GeminiTranslator
from .analyzer import CallAnalyzer
from .config import Config, load_config

__all__ = [
    "CallAnalysisPipeline",
    "SarvamTranscriber",
    "AudioCleaner",
    "SpeakerDiarizer",
    "GeminiTranslator",
    "CallAnalyzer",
    "Config",
    "load_config",
]
