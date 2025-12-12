"""
Standalone script to translate existing Hindi transcript using LLM (Gemini).
Requires no re-transcription.
"""
import os
import re
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(".env.local")

# Configuration
INPUT_FILE = r"D:\Skipper\Call-optimaization\transcripts\cleaned_Voice_hindi.txt"
OUTPUT_FILE = r"D:\Skipper\Call-optimaization\transcripts\cleaned_Voice_english_LLM.txt"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def parse_transcript(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    segments = []
    current_speaker = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("="):
            continue
            
        match = re.match(r"(Speaker \d+): (.+)", line)
        if match:
            segments.append({
                "speaker": match.group(1),
                "text": match.group(2)
            })
    return segments

def translate_with_gemini(segments):
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not found in .env.local")
        print("   Please get a free key from https://aistudio.google.com/app/apikey")
        return []

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    print("🤖 Sending to Gemini for context-aware translation...")
    
    # Prepare prompt
    transcript_text = ""
    for i, seg in enumerate(segments):
        transcript_text += f"[{i}] {seg['speaker']}: {seg['text']}\n"
        
    prompt = f"""
    You are an expert translator. Translate the following Hindi conversation to natural English.
    
    Rules:
    1. Maintain speaker tone and intent.
    2. Handle "Hinglish" (mixed language) naturally.
    3. Output format must be line-by-line corresponding to the input.
    4. Format: "[ID] SPEAKER: Translation"
    
    Transcript:
    {transcript_text}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return
        
    print(f"📂 Reading {INPUT_FILE}...")
    segments = parse_transcript(INPUT_FILE)
    print(f"✅ Parsed {len(segments)} lines.")
    
    translation = translate_with_gemini(segments)
    
    if translation:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("====================================================\n")
            f.write("ENGLISH TRANSCRIPT (High Quality LLM Translation)\n")
            f.write("====================================================\n\n")
            f.write(translation)
        
        print(f"💾 Saved high-quality translation to:\n   {OUTPUT_FILE}")
    else:
        print("❌ No translation generated.")

if __name__ == "__main__":
    main()
