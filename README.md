# Voice Transcription Module

End-to-end call analysis pipeline: audio cleaning, speaker diarization, transcription (Sarvam AI), translation (Gemini), and call quality analysis.

Built for the Skipper Pipes call quality system. Automates transcription and scoring, replacing manual QA review.

## Pipeline

```
Audio File (.mp3/.wav)
       │
       ▼
┌──────────────┐
│ Audio Clean   │  Noise reduction, normalization
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Diarization   │  pyannote.audio speaker segmentation
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Transcription │  Sarvam AI speech-to-text (Hindi/English)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Translation   │  Gemini LLM context-aware translation
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Analysis      │  Call quality scoring, summary, insights
└──────────────┘
```

## Quick Start

```bash
pip install -r voice_module/requirements.txt
```

```python
from voice_module import CallAnalysisPipeline
from pathlib import Path

pipeline = CallAnalysisPipeline()
results = pipeline.process(Path("call_recording.mp3"))

print(results["transcript"])
print(results["analysis"])
```

Or run directly:

```bash
python -m voice_module call_recording.mp3
```

## Configuration

Set environment variables (or use `.env`):

```bash
SARVAM_API_KEY=your_sarvam_key
GEMINI_API_KEY=your_gemini_key
HUGGINGFACE_TOKEN=your_hf_token
```

## Modules

| Module | Purpose |
|--------|---------|
| `pipeline.py` | End-to-end pipeline orchestrator |
| `audio_processing.py` | Noise reduction, normalization |
| `diarization.py` | Speaker segmentation (pyannote.audio) |
| `transcription.py` | Speech-to-text (Sarvam AI) |
| `translation.py` | Translation (Google Gemini) |
| `analyzer.py` | Call quality analysis and scoring |
| `config.py` | Configuration management |
| `validation.py` | Input validation |
| `retry_utils.py` | Retry with exponential backoff |
| `noise_reducer.py` | Standalone noise reduction |

## Dependencies

- `sarvamai` - Sarvam AI transcription API
- `google-generativeai` - Gemini translation
- `pyannote.audio` - Speaker diarization
- `torch` - PyTorch (for pyannote)
- `librosa`, `soundfile` - Audio processing
- `noisereduce` - Noise reduction
- `scipy` - Signal processing
