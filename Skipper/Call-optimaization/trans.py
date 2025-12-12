"""
Production-Ready Call Transcription System
================================================
Dependencies (compatible versions):
- Python >= 3.10
- pyannote.audio >= 3.1.0
- torch >= 2.0.0
- torchaudio >= 2.0.0
- sarvamai (latest)
- httpx == 0.24.1 (for sarvamai compatibility)

Features:
- Pyannote.audio for state-of-the-art speaker diarization
- Sarvam AI for Hindi transcription
- Sarvam AI for Hindi→English translation with chunking
- Optimized for efficiency and accuracy
"""

import os
import json
import time
import re
import shutil
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Core dependencies
import torch
import torchaudio
from pyannote.audio import Pipeline
from sarvamai import SarvamAI
from deep_translator import GoogleTranslator

# Configuration
SARVAM_API_KEY = "sk_gv7tk1oz_tEqghcgXEXaXXY4ukj2lEokP"
HUGGINGFACE_TOKEN = "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"  # Get from hf.co/settings/tokens
INPUT_AUDIO_PATH = r"D:\Skipper\Call-optimaization\input_1\cleaned_Voice.wav"
OUTPUT_DIR = r"D:\Skipper\Call-optimaization\transcripts"

# Optimization settings
USE_GPU = torch.cuda.is_available()
BATCH_TRANSCRIPTION = True  # Process multiple segments together
MIN_SEGMENT_DURATION = 0.5  # Skip segments shorter than 0.5s
MERGE_GAP_THRESHOLD = 1.0  # Merge segments with gaps < 1s


class GoogleTranslatorHandler:
    """
    Optimized translation handler using Google Translate.
    """
    
    def __init__(self):
        self.max_chunk_size = 4500  # Google Translate has ~5000 char limit
        self.cache = {}  # Simple translation cache
        
    def _create_chunks(self, text: str) -> List[str]:
        """Optimized chunking with sentence boundary detection."""
        if len(text) <= self.max_chunk_size:
            return [text]
        
        # Split on Hindi and English sentence boundaries
        sentences = re.split(r'([।|.!?]+\s+)', text)
        
        # Recombine sentences with punctuation
        combined = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined.append((sentences[i] + sentences[i+1]).strip())
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            combined.append(sentences[-1].strip())
        
        chunks = []
        current = ""
        
        for sentence in combined:
            if not sentence: continue
            
            # Smart split for very long sentences if needed
            if len(sentence) > self.max_chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                # Force split
                for i in range(0, len(sentence), self.max_chunk_size):
                    chunks.append(sentence[i:i+self.max_chunk_size])
                continue
                
            if len(current) + len(sentence) + 1 <= self.max_chunk_size:
                current += sentence + " "
            else:
                chunks.append(current.strip())
                current = sentence + " "
        
        if current:
            chunks.append(current.strip())
            
        return chunks
    
    def translate_text(self, text: str, source="hi", target="en") -> Optional[str]:
        """Translate with chunking and caching using Google Translate."""
        if not text or not text.strip():
            return None
            
        if text in self.cache:
            return self.cache[text]
            
        chunks = self._create_chunks(text)
        translated_chunks = []
        
        if len(chunks) > 1:
            print(f"  📊 Splitting into {len(chunks)} chunks for Google Translate...")
            
        try:
            translator = GoogleTranslator(source=source, target=target)
            
            for chunk in chunks:
                translated = translator.translate(chunk)
                if translated:
                    translated_chunks.append(translated)
                time.sleep(0.2)  # Gentle rate limit
                
            result = " ".join(translated_chunks)
            self.cache[text] = result
            return result
            
        except Exception as e:
            print(f"❌ Translation failed: {e}")
            return None


class AudioProcessor:
    """Handles audio loading and segmentation."""
    
    @staticmethod
    def get_duration(audio_path: str) -> Optional[float]:
        """Get audio duration efficiently."""
        try:
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception as e:
            print(f"⚠️  Could not read duration: {e}")
            return None
    
    @staticmethod
    def extract_segments(
        audio_path: str, 
        segments: List[Dict], 
        output_dir: str
    ) -> List[Dict]:
        """Extract audio segments efficiently."""
        print(f"\n📂 Extracting {len(segments)} audio segments...")
        
        # Load audio once
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        audio_chunks = []
        os.makedirs(output_dir, exist_ok=True)
        
        for i, segment in enumerate(segments):
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            
            segment_audio = waveform[:, start_sample:end_sample]
            temp_path = os.path.join(output_dir, f"seg_{i:04d}.wav")
            
            torchaudio.save(temp_path, segment_audio, sample_rate)
            
            audio_chunks.append({
                'path': temp_path,
                'speaker': segment['speaker'],
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration']
            })
        
        print(f"✅ Extracted {len(audio_chunks)} segments")
        return audio_chunks


def perform_diarization(audio_path: str, num_speakers: int = 2) -> Optional[List[Dict]]:
    """
    Perform speaker diarization with pyannote.audio.
    Optimized for production use.
    """
    print("\n🎯 Initializing speaker diarization (Pyannote 3.1)...")
    
    try:
        # Load pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN
        )
        
        # Optimize for device
        if USE_GPU:
            device = torch.device("cuda")
            pipeline.to(device)
            print(f"⚡ Running on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("🔄 Running on CPU")
        
        # Run diarization
        print("🔄 Analyzing speakers...")
        diarization = pipeline(audio_path, num_speakers=num_speakers)
        
        # Extract segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            # Filter very short segments
            if duration >= MIN_SEGMENT_DURATION:
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': duration
                })
        
        print(f"✅ Found {len(segments)} speech segments from {num_speakers} speakers")
        return segments
        
    except Exception as e:
        print(f"❌ Diarization error: {str(e)}")
        print("\n💡 Setup checklist:")
        print("   1. Accept conditions at: hf.co/pyannote/speaker-diarization-3.1")
        print("   2. Accept conditions at: hf.co/pyannote/segmentation-3.0")
        print("   3. Create token at: hf.co/settings/tokens")
        print("   4. Install: pip install pyannote.audio torch torchaudio")
        return None


def transcribe_segments(
    client: SarvamAI, 
    audio_chunks: List[Dict]
) -> List[Dict]:
    """
    Transcribe audio segments with Sarvam AI.
    Optimized with error handling.
    """
    print(f"\n📝 Transcribing {len(audio_chunks)} segments...")
    
    transcribed = []
    failed = 0
    
    for i, chunk in enumerate(audio_chunks):
        print(f"  Segment {i+1}/{len(audio_chunks)} ({chunk['duration']:.1f}s)...", end=" ")
        
        try:
            with open(chunk['path'], 'rb') as audio_file:
                response = client.speech_to_text.transcribe(
                    file=audio_file,
                    model="saarika:v2.5",
                    language_code="hi-IN"
                )
            
            text = None
            if isinstance(response, dict):
                text = response.get("text") or response.get("transcript")
            else:
                text = getattr(response, "text", None) or getattr(response, "transcript", None)
            
            if text and text.strip():
                transcribed.append({
                    'speaker': chunk['speaker'],
                    'start': chunk['start'],
                    'end': chunk['end'],
                    'text': text.strip()
                })
                print("✅")
            else:
                failed += 1
                print("⚠️ Empty")
                
        except Exception as e:
            failed += 1
            print(f"❌ {str(e)[:40]}")
    
    print(f"\n✅ Transcribed: {len(transcribed)}, Failed: {failed}")
    return transcribed


def merge_consecutive_turns(segments: List[Dict]) -> List[Dict]:
    """
    Merge consecutive segments from same speaker.
    Optimized for natural conversation flow.
    """
    if not segments:
        return []
    
    merged = []
    current = segments[0].copy()
    
    for segment in segments[1:]:
        gap = segment['start'] - current['end']
        same_speaker = segment['speaker'] == current['speaker']
        
        # Merge if same speaker and gap is small
        if same_speaker and gap <= MERGE_GAP_THRESHOLD:
            current['end'] = segment['end']
            current['text'] += " " + segment['text']
        else:
            merged.append(current)
            current = segment.copy()
    
    merged.append(current)
    print(f"🔗 Merged into {len(merged)} conversation turns")
    return merged


def format_transcript(segments: List[Dict], label_format: str = "Speaker") -> str:
    """Format segments into readable transcript."""
    lines = []
    for seg in segments:
        speaker_label = seg['speaker'].replace('SPEAKER_', label_format + ' ')
        lines.append(f"{speaker_label}: {seg['text']}")
    return "\n\n".join(lines)


def save_transcript(content: str, path: str, title: str):
    """Save formatted transcript to file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 70 + "\n\n")
        f.write(content)
    print(f"💾 Saved: {Path(path).name}")


def main():
    """Main pipeline with optimized workflow."""
    print("=" * 70)
    print("🎤 Production Call Transcription System")
    print("   Pyannote Diarization → Sarvam Transcription → Translation")
    print("=" * 70)
    
    # Validate input
    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"❌ Audio file not found: {INPUT_AUDIO_PATH}")
        return
    
    audio_path = Path(INPUT_AUDIO_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Input: {audio_path.name}")
    
    # Check duration
    duration = AudioProcessor.get_duration(str(audio_path))
    if duration:
        print(f"⏱️  Duration: {duration:.1f}s ({duration/60:.1f}min)")
    
    # Initialize Sarvam client
    try:
        client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
        print("✅ Sarvam AI client initialized")
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        print("💡 Try: pip install httpx==0.24.1 sarvamai")
        return
    
    # Step 1: Diarization
    print("\n" + "="*70)
    print("STEP 1: SPEAKER DIARIZATION")
    print("="*70)
    segments = perform_diarization(str(audio_path), num_speakers=2)
    if not segments:
        return
    
    # Step 2: Extract segments
    print("\n" + "="*70)
    print("STEP 2: AUDIO SEGMENTATION")
    print("="*70)
    temp_dir = output_dir / "temp_segments"
    audio_chunks = AudioProcessor.extract_segments(str(audio_path), segments, str(temp_dir))
    
    # Step 3: Transcription
    print("\n" + "="*70)
    print("STEP 3: HINDI TRANSCRIPTION")
    print("="*70)
    transcribed = transcribe_segments(client, audio_chunks)
    if not transcribed:
        print("❌ No successful transcriptions")
        return
    
    # Merge consecutive turns
    merged = merge_consecutive_turns(transcribed)
    
    # Save Hindi transcript
    hindi_text = format_transcript(merged)
    hindi_path = output_dir / f"{audio_path.stem}_hindi.txt"
    save_transcript(hindi_text, str(hindi_path), "HINDI TRANSCRIPT (Pyannote + Sarvam)")
    
    print(f"\n📄 Preview:\n{hindi_text[:300]}...")
    
    # Step 4: Translation
    print("\n" + "="*70)
    print("STEP 4: ENGLISH TRANSLATION (Google)")
    print("="*70)
    
    translator = GoogleTranslatorHandler()
    
    english_segments = []
    for i, seg in enumerate(merged):
        print(f"\n{seg['speaker']} ({i+1}/{len(merged)}):")
        english_text = translator.translate_text(seg['text'])
        
        if english_text:
            english_segments.append({'speaker': seg['speaker'], 'text': english_text})
    
    # Save English transcript
    if english_segments:
        english_text = format_transcript(english_segments)
        english_path = output_dir / f"{audio_path.stem}_english.txt"
        save_transcript(english_text, str(english_path), "ENGLISH TRANSLATION")
        
        print(f"\n📄 Preview:\n{english_text[:300]}...")
    
    # Step 5: Bilingual transcript
    print("\n" + "="*70)
    print("STEP 5: BILINGUAL TRANSCRIPT")
    print("="*70)
    
    bilingual_path = output_dir / f"{audio_path.stem}_bilingual.txt"
    with open(bilingual_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BILINGUAL CALL TRANSCRIPT\n")
        f.write("=" * 80 + "\n\n")
        
        for i in range(len(merged)):
            speaker = merged[i]['speaker'].replace('SPEAKER_', 'Speaker ')
            f.write(f"[{speaker}]\n")
            f.write(f"Hindi:   {merged[i]['text']}\n")
            if i < len(english_segments):
                f.write(f"English: {english_segments[i]['text']}\n")
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"💾 Saved: {bilingual_path.name}")
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print("\n🧹 Cleaned up temporary files")
    
    # Summary
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE")
    print("="*70)
    print(f"📊 Statistics:")
    print(f"   • Speakers: 2")
    print(f"   • Segments: {len(segments)} → {len(merged)} turns")
    print(f"   • Duration: {duration:.1f}s")
    print(f"\n📁 Output files:")
    print(f"   • {hindi_path.name}")
    print(f"   • {english_path.name}")
    print(f"   • {bilingual_path.name}")
    print("="*70)


if __name__ == "__main__":
    main()