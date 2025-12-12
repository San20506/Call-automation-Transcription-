---
description: Complete call analysis pipeline - noisy audio to analysis reports
---

# Call Analysis Workflow

Processes noisy call recordings through cleaning, transcription, and Gemini analysis to generate structured outputs.

## Prerequisites
- Python environment with dependencies installed (`pip install -r requirements.txt`)
- `.env.local` file with API keys:
  - `SARVAM_API_KEY` - For Hindi transcription
  - `GEMINI_API_KEY` - For translation and analysis
  - `HUGGINGFACE_TOKEN` - For speaker diarization (optional if cached)

---

## Quick Run (Single File)

// turbo
```powershell
python d:\Skipper\Call-optimaization\call_analyzer.py --input "path\to\audio.mp3"
```

---

## Pipeline Steps

### 1. Noise Reduction
- Removes stationary noise (background hum)
- Removes non-stationary noise (traffic, voices)
- Filters low/high frequency artifacts
- Normalizes audio levels

### 2. Speaker Diarization
- Identifies different speakers in the call
- Uses Pyannote.audio for accurate segmentation
- Requires HuggingFace token for first run

### 3. Transcription
- Transcribes Hindi/Hinglish audio via Sarvam AI
- Processes each speaker segment individually

### 4. Translation
- Gemini 1.5 Flash for contextual translation
- Handles Hinglish mixed content

### 5. Analysis (Gemini)
Generates 3 output files:

| Output | Description |
|--------|-------------|
| `_CONTEXT.txt` | Call purpose, product discussed, customer type |
| `_SUMMARY.txt` | Key points, agreements, outcomes |
| `_EFFICIENCY.txt` | Salesperson mistakes & improvement suggestions |

---

## Outputs

Files are saved to `D:\Skipper\Call-optimaization\output\`:

```
{filename}_{timestamp}_CONTEXT.txt
{filename}_{timestamp}_SUMMARY.txt
{filename}_{timestamp}_EFFICIENCY.txt
{filename}_{timestamp}_TRANSCRIPT.txt (bilingual)
```

---

## Batch Processing

To process multiple files:

```python
from call_analyzer import run_pipeline
import glob

for audio in glob.glob("d:/Skipper/Call-optimaization/input/*.mp3"):
    run_pipeline(audio)
```

---

## Context

**Call Type:** Company calling customers
- **Caller:** Salesperson
- **Receiver:** Customer
