"""
Production-Ready Call Transcription System with Advanced LLM Translation
========================================================================
Features:
- Pyannote.audio for state-of-the-art speaker diarization
- Sarvam AI for Hindi transcription
- Google Gemini Pro (or OpenAI GPT-4) for context-aware translation
- Optimized for efficiency and accuracy

Dependencies:
Check requirements.txt
"""

import os
import json
import time
import shutil
import logging
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Core dependencies
import torch
import torchaudio
import soundfile as sf
from pyannote.audio import Pipeline
from sarvamai import SarvamAI
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(".env.local")

# Configuration
CONFIG = {
    "SARVAM_API_KEY": os.getenv("SARVAM_API_KEY", "sk_gv7tk1oz_tEqghcgXEXaXXY4ukj2lEokP"),
    "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN", "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"),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),  # Needs to be set for better translation
    "INPUT_AUDIO_PATH": r"D:\Skipper\Call-optimaization\input_1\cleaned_Voice.wav",
    "OUTPUT_DIR": r"D:\Skipper\Call-optimaization\transcripts",
    "USE_GPU": torch.cuda.is_available(),
    "MIN_SEGMENT_DURATION": 0.5,
    "MERGE_GAP_THRESHOLD": 1.0,
}


class LLMTranslator:
    """
    Advanced translation using Google Gemini Pro.
    Better than Google Translate for Hinglish and context.
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API Key is missing. Set GEMINI_API_KEY.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.cache = {}

    def translate_contextual(self, segments: List[Dict]) -> List[Dict]:
        """
        Translates entire conversation with context awareness.
        Sending batches makes it understand speaker flow.
        """
        logger.info("🤖 Starting LLM Contextual Translation...")
        
        # Prepare the transcript context
        full_transcript = "Conversation Transcript:\n"
        for i, seg in enumerate(segments):
            full_transcript += f"[{i}] {seg['speaker']}: {seg['text']}\n"
            
        prompt = f"""
        You are an expert translator specializing in Hindi/Hinglish to English spoken conversations.
        Translate the following conversation to natural, professional English.
        
        Rules:
        1. Preserve the speaker labels and IDs exactly.
        2. Handle "Hinglish" (mixed Hindi/English) naturally.
        3. Output MUST be valid JSON list matching the input structure size.
        4. If a sentence is already English, polish it.
        
        Input Format:
        [ID] SPEAKER: Text
        
        Make sure to return a JSON array of strings corresponding to the translations of each line ID in order.
        Example: ["Translation 1", "Translation 2"]
        
        {full_transcript}
        """
        
        try:
            # We might need to split if too long, but Gemini 1.5 has huge context window
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            
            translations = json.loads(response.text)
            
            if len(translations) != len(segments):
                logger.warning(f"Mismatch in translation count: Got {len(translations)}, expected {len(segments)}")
                # Fallback to simple mapping if length mismatch (rare)
                return self._fallback_translate_individually(segments)
            
            # Map back
            for i, text in enumerate(translations):
                segments[i]['english_text'] = text
                
            return segments
            
        except Exception as e:
            logger.error(f"LLM Batch Translation failed: {e}")
            logger.info("Falling back to individual translation...")
            return self._fallback_translate_individually(segments)

    def _fallback_translate_individually(self, segments: List[Dict]) -> List[Dict]:
        for seg in segments:
            seg['english_text'] = self.translate_text(seg['text'])
        return segments

    def translate_text(self, text: str) -> str:
        if text in self.cache: return self.cache[text]
        try:
            response = self.model.generate_content(
                f"Translate this Hindi/Hinglish spoken text to English. Output only the translation: '{text}'"
            )
            result = response.text.strip()
            self.cache[text] = result
            return result
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original if fail


class AudioProcessor:
    @staticmethod
    def extract_segments(audio_path: str, segments: List[Dict], output_dir: str) -> List[Dict]:
        logger.info(f"📂 Extracting {len(segments)} audio segments...")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        audio_chunks = []
        os.makedirs(output_dir, exist_ok=True)
        
        for i, segment in enumerate(segments):
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            
            segment_audio = waveform[:, start_sample:end_sample]
            temp_path = os.path.join(output_dir, f"seg_{i:04d}.wav")
            sf.write(temp_path, segment_audio.squeeze().numpy(), sample_rate)
            
            audio_chunks.append({
                'path': temp_path,
                'speaker': segment['speaker'],
                'start': segment['start'],
                'end': segment['end']
            })
        return audio_chunks


def perform_diarization(audio_path: str) -> List[Dict]:
    logger.info("🎯 Initializing Pyannote Diarization...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=CONFIG["HUGGINGFACE_TOKEN"]
        )
        if CONFIG["USE_GPU"]:
            pipeline.to(torch.device("cuda"))
        
        diarization = pipeline(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if (turn.end - turn.start) >= CONFIG["MIN_SEGMENT_DURATION"]:
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
        logger.info(f"✅ Found {len(segments)} segments")
        return segments
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return []

def transcribe_segments(client: SarvamAI, chunks: List[Dict]) -> List[Dict]:
    logger.info("📝 Transcribing with Sarvam AI...")
    transcribed = []
    
    for i, chunk in enumerate(chunks):
        try:
            with open(chunk['path'], 'rb') as f:
                # Assuming Sarvam has speech_to_text_translate or similar. 
                # Using generic transcribe for now as per previous working code.
                response = client.speech_to_text.transcribe(
                    file=f,
                    model="saarika:v2.5",
                    language_code="hi-IN"
                )
            # Handle response extraction (adjust based on actual API response structure)
            text = response.text if hasattr(response, 'text') else response.get('text', '')
            
            if text:
                transcribed.append({**chunk, 'text': text})
                print(".", end="", flush=True)
        except Exception as e:
            logger.error(f"Error seg {i}: {e}")
            
    print("\n")
    return transcribed

def merge_turns(segments: List[Dict]) -> List[Dict]:
    if not segments: return []
    merged = []
    current = segments[0].copy()
    
    for seg in segments[1:]:
        if (seg['speaker'] == current['speaker'] and 
            (seg['start'] - current['end']) < CONFIG['MERGE_GAP_THRESHOLD']):
            current['end'] = seg['end']
            current['text'] += " " + seg['text']
        else:
            merged.append(current)
            current = seg.copy()
    merged.append(current)
    return merged

def save_outputs(segments: List[Dict], base_name: str):
    timestamp = time.strftime("%Y%m%d-%H%M")
    out_dir = Path(CONFIG["OUTPUT_DIR"])
    
    # Bilingual
    with open(out_dir / f"{base_name}_FULL_{timestamp}.txt", "w", encoding="utf-8") as f:
         for seg in segments:
             speaker = seg['speaker'].replace("SPEAKER_", "Speaker ")
             f.write(f"[{speaker}]\n")
             f.write(f"Hindi:   {seg['text']}\n")
             f.write(f"English: {seg.get('english_text', '')}\n\n")

def main():
    if not os.path.exists(CONFIG["INPUT_AUDIO_PATH"]):
        logger.error("Input file not found")
        return

    # 1. Diarization
    segments = perform_diarization(CONFIG["INPUT_AUDIO_PATH"])
    if not segments: return

    # 2. Extract & Transcribe
    temp_dir = Path(CONFIG["OUTPUT_DIR"]) / "temp_segments"
    chunks = AudioProcessor.extract_segments(CONFIG["INPUT_AUDIO_PATH"], segments, str(temp_dir))
    
    sarvam_client = SarvamAI(api_subscription_key=CONFIG["SARVAM_API_KEY"])
    transcribed_segments = transcribe_segments(sarvam_client, chunks)
    
    # 3. Merge
    merged_segments = merge_turns(transcribed_segments)
    
    # 4. Smart Translation (LLM)
    # Check if user wants to use Gemini or fallback
    if CONFIG["GEMINI_API_KEY"]:
        translator = LLMTranslator(CONFIG["GEMINI_API_KEY"])
        final_segments = translator.translate_contextual(merged_segments)
    else:
        logger.warning("No GEMINI_API_KEY found. Skipping advanced translation.")
        final_segments = merged_segments # Or fallback to Google Translate

    # 5. Save
    save_outputs(final_segments, Path(CONFIG["INPUT_AUDIO_PATH"]).stem)
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
