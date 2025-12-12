# Production Call Transcription System - Setup Guide

## 📋 System Requirements

- **Python**: 3.10 or higher (3.11 recommended)
- **OS**: Windows, Linux, or macOS
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended (NVIDIA for PyTorch acceleration)

## 🚀 Installation Steps

1.  **Install Dependencies**
    Run the following command in your terminal to install all required packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Hugging Face Setup (Crucial for Diarization)**
    - Pyannote.audio requires you to accept user conditions.
    - Go to [hf.co/pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) and accept the terms.
    - Go to [hf.co/pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) and accept the terms.
    - Create a user Access Token at [hf.co/settings/tokens](https://hf.co/settings/tokens) (Read premissions are sufficient).
    - Update `HUGGINGFACE_TOKEN` in `trans.py` with your token.

3.  **Sarvam AI Setup**
    - Ensure you have a valid API key from Sarvam AI.
    - Update `SARVAM_API_KEY` in `trans.py` (or set it in `.env.local` if using environment variables).

## 🏃‍♂️ Usage

Run the main transcription script:
```bash
python trans.py
```

The script will:
1.  Perform Speaker Diarization (identifying who is speaking).
2.  Segment the audio.
3.  Transcribe each segment using Sarvam AI (Hindi).
4.  Translate the Hindi text to English using Google Translate (deep-translator).
5.  Generate 3 output files in the `transcripts` folder:
    - `*_hindi.txt`
    - `*_english.txt`
    - `*_bilingual.txt`

## ⚠️ Common Issues

- **Dependency Conflicts**: If you see `httpx` errors, ensure you have exactly version 0.24.1:
  `pip install httpx==0.24.1`
- **Authentication Error**: Check your Hugging Face token and unsure you accepted the model terms on the website.
