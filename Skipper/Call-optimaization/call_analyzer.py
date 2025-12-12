"""
Call Analysis Pipeline
======================
Complete workflow: Noisy Audio → Clean → Transcribe → Analyze → Reports

Outputs:
1. Call Context - What the call is about
2. Call Summary - Key points and outcomes  
3. Efficiency Report - Salesperson performance analysis

Context: Company calling customers (Caller = Salesperson, Receiver = Customer)
"""

import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Audio processing
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy import signal

# Transcription
import torch
import torchaudio
from pyannote.audio import Pipeline
from sarvamai import SarvamAI

# LLM Analysis
import google.generativeai as genai

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv(".env.local")

CONFIG = {
    "SARVAM_API_KEY": os.getenv("SARVAM_API_KEY"),
    "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN", "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"),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "INPUT_DIR": r"D:\Skipper\Call-optimaization\input",
    "OUTPUT_DIR": r"D:\Skipper\Call-optimaization\output",
    "TEMP_DIR": r"D:\Skipper\Call-optimaization\temp",
    "USE_GPU": torch.cuda.is_available(),
    "MIN_SEGMENT_DURATION": 0.5,
    "MERGE_GAP_THRESHOLD": 1.0,
}


class NoiseReducer:
    """Multi-stage audio cleaning for noisy call recordings."""
    
    @staticmethod
    def clean_audio(input_file: str, output_dir: str) -> str:
        """Clean noisy audio with multi-stage noise reduction."""
        logger.info(f"🔇 Cleaning audio: {os.path.basename(input_file)}")
        
        # Load audio
        y, sr = librosa.load(input_file, sr=None)
        logger.info(f"  Duration: {len(y)/sr:.1f}s | Sample rate: {sr}Hz")
        
        # Stage 1: Remove stationary noise
        logger.info("  [1/4] Removing stationary noise...")
        y_clean = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.90)
        
        # Stage 2: Remove non-stationary noise (traffic, etc)
        logger.info("  [2/4] Removing non-stationary noise...")
        y_clean = nr.reduce_noise(y=y_clean, sr=sr, stationary=False, prop_decrease=0.70)
        
        # Stage 3: Highpass filter (remove low-frequency rumble)
        logger.info("  [3/4] Filtering frequencies...")
        highpass_freq = min(80, sr * 0.01)
        sos_high = signal.butter(4, highpass_freq, 'hp', fs=sr, output='sos')
        y_clean = signal.sosfilt(sos_high, y_clean)
        
        # Lowpass filter (remove hiss)
        nyquist = sr / 2
        lowpass_freq = min(nyquist * 0.95, 8000)
        sos_low = signal.butter(4, lowpass_freq, 'lp', fs=sr, output='sos')
        y_clean = signal.sosfilt(sos_low, y_clean)
        
        # Stage 4: Normalize
        logger.info("  [4/4] Normalizing audio...")
        peak = np.abs(y_clean).max()
        if peak > 0:
            y_clean = y_clean / peak * 0.707
        y_clean = np.clip(y_clean * 1.5, -1.0, 1.0)
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"cleaned_{Path(input_file).stem}.wav")
        sf.write(output_file, y_clean, sr, subtype='PCM_16')
        
        logger.info(f"  ✓ Cleaned audio saved: {output_file}")
        return output_file


class Transcriber:
    """Speaker diarization + Hindi transcription + English translation."""
    
    def __init__(self):
        self.device = torch.device("cuda" if CONFIG["USE_GPU"] else "cpu")
        self.sarvam = SarvamAI(api_subscription_key=CONFIG["SARVAM_API_KEY"])
        self._init_diarization()
        
    def _init_diarization(self):
        """Initialize Pyannote speaker diarization."""
        logger.info("🎯 Loading speaker diarization model...")
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=CONFIG["HUGGINGFACE_TOKEN"]
            )
            self.pipeline.to(self.device)
        except Exception as e:
            logger.error(f"Diarization init failed: {e}")
            raise
            
    def process_audio(self, audio_path: str, temp_dir: str) -> List[Dict]:
        """Full transcription pipeline."""
        # 1. Diarization
        segments = self._diarize(audio_path)
        
        # 2. Extract segments
        chunks = self._extract_segments(audio_path, segments, temp_dir)
        
        # 3. Transcribe each segment
        transcribed = self._transcribe(chunks)
        
        # 4. Merge consecutive same-speaker segments
        merged = self._merge_turns(transcribed)
        
        return merged
    
    def _diarize(self, audio_path: str) -> List[Dict]:
        """Identify speakers and their segments."""
        logger.info("🎤 Running speaker diarization...")
        diarization = self.pipeline(audio_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if (turn.end - turn.start) >= CONFIG["MIN_SEGMENT_DURATION"]:
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
        logger.info(f"  ✓ Found {len(segments)} speaker segments")
        return segments
    
    def _extract_segments(self, audio_path: str, segments: List[Dict], output_dir: str) -> List[Dict]:
        """Extract individual audio chunks per segment."""
        logger.info(f"📂 Extracting {len(segments)} audio segments...")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        os.makedirs(output_dir, exist_ok=True)
        chunks = []
        
        for i, seg in enumerate(segments):
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            segment_audio = waveform[:, start_sample:end_sample]
            
            path = os.path.join(output_dir, f"seg_{i:04d}.wav")
            sf.write(path, segment_audio.squeeze().numpy(), sample_rate)
            chunks.append({**seg, 'path': path})
            
        return chunks
    
    def _transcribe(self, chunks: List[Dict]) -> List[Dict]:
        """Transcribe each audio chunk with Sarvam AI."""
        logger.info("📝 Transcribing audio segments...")
        transcribed = []
        
        for i, chunk in enumerate(chunks):
            try:
                with open(chunk['path'], 'rb') as f:
                    response = self.sarvam.speech_to_text.transcribe(
                        file=f,
                        model="saarika:v2.5",
                        language_code="hi-IN"
                    )
                text = response.text if hasattr(response, 'text') else response.get('text', '')
                
                if text:
                    transcribed.append({**chunk, 'text': text})
                    print(".", end="", flush=True)
            except Exception as e:
                logger.error(f"Transcription error for segment {i}: {e}")
                
        print()
        logger.info(f"  ✓ Transcribed {len(transcribed)} segments")
        return transcribed
    
    def _merge_turns(self, segments: List[Dict]) -> List[Dict]:
        """Merge consecutive segments from same speaker."""
        if not segments:
            return []
            
        merged = []
        current = segments[0].copy()
        
        for seg in segments[1:]:
            gap = seg['start'] - current['end']
            if seg['speaker'] == current['speaker'] and gap < CONFIG['MERGE_GAP_THRESHOLD']:
                current['end'] = seg['end']
                current['text'] += " " + seg['text']
            else:
                merged.append(current)
                current = seg.copy()
        merged.append(current)
        
        return merged


class CallAnalyzer:
    """Gemini-powered call analysis for context, summary, and efficiency."""
    
    def __init__(self):
        if not CONFIG["GEMINI_API_KEY"]:
            raise ValueError("GEMINI_API_KEY is required for analysis")
        genai.configure(api_key=CONFIG["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def analyze(self, transcript_segments: List[Dict]) -> Dict:
        """Perform full analysis and return structured results."""
        logger.info("🤖 Running Gemini analysis...")
        
        # Build conversation text
        conversation = self._build_conversation_text(transcript_segments)
        
        # Run analysis
        results = {
            'context': self._get_context(conversation),
            'summary': self._get_summary(conversation),
            'efficiency': self._get_efficiency(conversation)
        }
        
        logger.info("  ✓ Analysis complete")
        return results
    
    def _build_conversation_text(self, segments: List[Dict]) -> str:
        """Format segments into readable conversation."""
        lines = []
        for seg in segments:
            speaker = seg['speaker'].replace("SPEAKER_", "Speaker ")
            lines.append(f"[{speaker}]: {seg['text']}")
        return "\n".join(lines)
    
    def _get_context(self, conversation: str) -> str:
        """Extract call context."""
        prompt = f"""Analyze this sales call conversation and provide the CONTEXT of the call.

CONTEXT: This is a Company calling its customers. The caller is a SALESPERSON and the receiver is a CUSTOMER.

Conversation:
{conversation}

Provide a brief context covering:
1. What product/service is being discussed
2. Purpose of the call (follow-up, feedback, survey, upselling, etc.)
3. Customer profile (if discernible)

Keep it concise (3-5 bullet points)."""

        return self._call_gemini(prompt)
    
    def _get_summary(self, conversation: str) -> str:
        """Generate call summary."""
        prompt = f"""Analyze this sales call conversation and provide a SUMMARY.

CONTEXT: This is a Company calling its customers. The caller is a SALESPERSON and the receiver is a CUSTOMER.

Conversation:
{conversation}

Provide a summary covering:
1. Key discussion points
2. Customer responses and feedback
3. Any agreements or next steps
4. Outcome of the call

Keep it concise but comprehensive."""

        return self._call_gemini(prompt)
    
    def _get_efficiency(self, conversation: str) -> str:
        """Analyze salesperson efficiency."""
        prompt = f"""Analyze this sales call from a SALESPERSON PERFORMANCE perspective.

CONTEXT: This is a Company calling its customers. The caller is a SALESPERSON and the receiver is a CUSTOMER.

Conversation:
{conversation}

Provide a brief efficiency report covering:

1. MISTAKES MADE:
   - Communication errors
   - Missed information
   - Poor handling of objections
   - Lack of clarity

2. WHAT COULD BE DONE BETTER:
   - Specific improvements for next time
   - Better approaches to use
   - Opportunities missed

Keep it SHORT and ACTIONABLE (focus on practical improvements)."""

        return self._call_gemini(prompt)
    
    def _call_gemini(self, prompt: str) -> str:
        """Make Gemini API call."""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"[Error generating analysis: {e}]"


def run_pipeline(input_file: str) -> Dict:
    """
    Complete pipeline: Clean → Transcribe → Translate → Analyze
    
    Args:
        input_file: Path to noisy audio file
        
    Returns:
        Dictionary with transcript and analysis results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_file).stem
    
    # Prepare directories
    temp_dir = os.path.join(CONFIG["TEMP_DIR"], timestamp)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Step 1: Clean audio
        logger.info("=" * 60)
        logger.info("STEP 1: NOISE REDUCTION")
        logger.info("=" * 60)
        cleaned_audio = NoiseReducer.clean_audio(input_file, temp_dir)
        
        # Step 2: Transcribe
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: TRANSCRIPTION & DIARIZATION")
        logger.info("=" * 60)
        transcriber = Transcriber()
        segments = transcriber.process_audio(cleaned_audio, os.path.join(temp_dir, "segments"))
        
        # Step 3: Translate with Gemini
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: TRANSLATION")
        logger.info("=" * 60)
        segments = _translate_segments(segments)
        
        # Step 4: Analyze
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: CALL ANALYSIS")
        logger.info("=" * 60)
        analyzer = CallAnalyzer()
        analysis = analyzer.analyze(segments)
        
        # Step 5: Save outputs
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: SAVING OUTPUTS")
        logger.info("=" * 60)
        output_paths = _save_outputs(base_name, timestamp, segments, analysis)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("=" * 60)
        for name, path in output_paths.items():
            logger.info(f"  {name}: {path}")
            
        return {
            'transcript': segments,
            'analysis': analysis,
            'outputs': output_paths
        }
        
    finally:
        # Cleanup temp files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def _translate_segments(segments: List[Dict]) -> List[Dict]:
    """Translate Hindi text to English using Gemini."""
    logger.info("🌐 Translating to English...")
    
    if not CONFIG["GEMINI_API_KEY"]:
        logger.warning("No GEMINI_API_KEY - skipping translation")
        return segments
        
    genai.configure(api_key=CONFIG["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Build batch for contextual translation
    full_transcript = "Translate this Hindi/Hinglish conversation to English:\n"
    for i, seg in enumerate(segments):
        full_transcript += f"[{i}] {seg['speaker']}: {seg['text']}\n"
    
    prompt = f"""{full_transcript}

Return a JSON array of translations in order, one for each line.
Example: ["Translation 1", "Translation 2"]
Output ONLY the JSON array, nothing else."""

    try:
        response = model.generate_content(
            prompt, 
            generation_config={"response_mime_type": "application/json"}
        )
        translations = json.loads(response.text)
        
        if len(translations) == len(segments):
            for i, text in enumerate(translations):
                segments[i]['english_text'] = text
        else:
            logger.warning("Translation count mismatch, falling back to original")
            
    except Exception as e:
        logger.error(f"Translation error: {e}")
        
    logger.info(f"  ✓ Translated {len(segments)} segments")
    return segments


def _save_outputs(base_name: str, timestamp: str, segments: List[Dict], analysis: Dict) -> Dict:
    """Save all output files."""
    output_dir = CONFIG["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    
    prefix = f"{base_name}_{timestamp}"
    paths = {}
    
    # 1. Call Context
    context_path = os.path.join(output_dir, f"{prefix}_CONTEXT.txt")
    with open(context_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("CALL CONTEXT\n")
        f.write("=" * 60 + "\n\n")
        f.write(analysis['context'])
    paths['Context'] = context_path
    
    # 2. Call Summary
    summary_path = os.path.join(output_dir, f"{prefix}_SUMMARY.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("CALL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(analysis['summary'])
    paths['Summary'] = summary_path
    
    # 3. Efficiency Report
    efficiency_path = os.path.join(output_dir, f"{prefix}_EFFICIENCY.txt")
    with open(efficiency_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("EFFICIENCY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(analysis['efficiency'])
    paths['Efficiency'] = efficiency_path
    
    # 4. Full Transcript (bonus)
    transcript_path = os.path.join(output_dir, f"{prefix}_TRANSCRIPT.txt")
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("FULL TRANSCRIPT (Bilingual)\n")
        f.write("=" * 60 + "\n\n")
        for seg in segments:
            speaker = seg['speaker'].replace("SPEAKER_", "Speaker ")
            f.write(f"[{speaker}]\n")
            f.write(f"Hindi:   {seg['text']}\n")
            f.write(f"English: {seg.get('english_text', seg['text'])}\n")
            f.write("-" * 40 + "\n")
    paths['Transcript'] = transcript_path
    
    return paths


def main():
    """Main entry point with CLI support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Call Analysis Pipeline")
    parser.add_argument('--input', '-i', required=True, help="Path to audio file")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return
        
    run_pipeline(args.input)


if __name__ == "__main__":
    main()
