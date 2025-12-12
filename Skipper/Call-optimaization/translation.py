"""
Translation Module
=================
Handles translation using Google Gemini with context awareness and caching.
"""

import logging
import json
from typing import List, Dict, Optional
from pathlib import Path
import google.generativeai as genai

from config import TranslationConfig
from validation import ValidationError
from retry_utils import retry_on_exception

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Raised when translation fails."""
    pass


class GeminiTranslator:
    """Context-aware translation using Google Gemini."""
    
    def __init__(self, 
                 api_key: str,
                 config: Optional[TranslationConfig] = None):
        """
        Initialize translator.
        
        Args:
            api_key: Google Gemini API key
            config: Optional translation configuration
            
        Raises:
            ValidationError: If API key is invalid
        """
        if not api_key:
            raise ValidationError("Gemini API key is required")
        
        self.api_key = api_key
        self.config = config or TranslationConfig()
        self.cache = {} if self.config.enable_caching else None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Gemini AI client."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.config.model)
            logger.info(f"✓ Gemini {self.config.model} initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise TranslationError(f"Client initialization failed: {e}") from e
    
    @retry_on_exception(
        exceptions=(ConnectionError, TimeoutError),
        max_attempts=3,
        base_delay=2.0,
        max_delay=30.0
    )
    def translate_text(self, 
                      text: str,
                      source_lang: Optional[str] = None,
                      target_lang: Optional[str] = None) -> str:
        """
        Translate single text string.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
            
        Raises:
            TranslationError: If translation fails
        """
        if not text or not text.strip():
            return ""
        
        # Check cache
        if self.cache is not None and text in self.cache:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return self.cache[text]
        
        source_lang = source_lang or self.config.source_language
        target_lang = target_lang or self.config.target_language
        
        prompt = f"""Translate the following {source_lang} text to {target_lang}.
Handle mixed language (Hinglish) naturally.
Output only the translation, nothing else.

Text: {text}"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens
                }
            )
            
            translated = response.text.strip()
            
            # Cache result
            if self.cache is not None:
                self.cache[text] = translated
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Failed to translate: {e}") from e
    
    @retry_on_exception(
        exceptions=(ConnectionError, TimeoutError),
        max_attempts=3,
        base_delay=2.0,
        max_delay=30.0
    )
    def translate_conversation(self, 
                              segments: List[Dict],
                              preserve_speakers: bool = True) -> List[Dict]:
        """
        Translate entire conversation with context awareness.
        
        Args:
            segments: List of conversation segments with 'text' field
            preserve_speakers: Keep speaker labels intact
            
        Returns:
            Segments with 'translated_text' field added
        """
        if not segments:
            return []
        
        logger.info(f"Translating conversation ({len(segments)} segments)...")
        
        # Build conversation context
        conversation = self._build_conversation_text(segments, preserve_speakers)
        
        # Create contextual translation prompt
        prompt = self._create_translation_prompt(conversation, preserve_speakers)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens
                }
            )
            
            # Parse JSON response
            translations = json.loads(response.text)
            
            # Validate response length
            if len(translations) != len(segments):
                logger.warning(
                    f"Translation count mismatch: got {len(translations)}, "
                    f"expected {len(segments)}. Falling back to individual translation."
                )
                return self._translate_individually(segments)
            
            # Add translations to segments
            translated_segments = []
            for segment, translation in zip(segments, translations):
                seg_copy = segment.copy()
                seg_copy['translated_text'] = translation
                seg_copy['target_language'] = self.config.target_language
                translated_segments.append(seg_copy)
            
            logger.info(f"✓ Translated {len(translated_segments)} segments")
            return translated_segments
            
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            logger.info("Falling back to individual translation...")
            return self._translate_individually(segments)
    
    def _build_conversation_text(self, 
                                segments: List[Dict],
                                preserve_speakers: bool) -> str:
        """Build formatted conversation text."""
        lines = []
        for i, seg in enumerate(segments):
            if preserve_speakers and 'speaker' in seg:
                speaker = seg['speaker'].replace('SPEAKER_', 'Speaker ')
                lines.append(f"[{i}] {speaker}: {seg['text']}")
            else:
                lines.append(f"[{i}] {seg['text']}")
        return "\n".join(lines)
    
    def _create_translation_prompt(self, 
                                  conversation: str,
                                  preserve_speakers: bool) -> str:
        """Create translation prompt with context."""
        speaker_instruction = (
            "Preserve speaker labels exactly as shown."
            if preserve_speakers else ""
        )
        
        return f"""You are an expert translator for Hindi/Hinglish to English conversations.

Task: Translate the following conversation to natural, professional English.

Rules:
1. Handle mixed Hindi/English (Hinglish) naturally
2. Maintain conversational tone
3. {speaker_instruction}
4. Output MUST be a JSON array of strings, one translation per line
5. Maintain the same order and count as the input

Example output format: ["Translation 1", "Translation 2", "Translation 3"]

Conversation:
{conversation}

Output only the JSON array, nothing else."""
    
    def _translate_individually(self, segments: List[Dict]) -> List[Dict]:
        """Fallback: translate segments individually."""
        logger.info("Translating segments individually...")
        
        translated_segments = []
        for segment in segments:
            try:
                text = segment.get('text', '')
                if text:
                    translation = self.translate_text(text)
                    seg_copy = segment.copy()
                    seg_copy['translated_text'] = translation
                    seg_copy['target_language'] = self.config.target_language
                    translated_segments.append(seg_copy)
            except Exception as e:
                logger.error(f"Failed to translate segment: {e}")
                # Add segment with original text
                seg_copy = segment.copy()
                seg_copy['translated_text'] = segment.get('text', '')
                translated_segments.append(seg_copy)
        
        return translated_segments
    
    def clear_cache(self) -> None:
        """Clear translation cache."""
        if self.cache is not None:
            cache_size = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared {cache_size} cached translations")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if self.cache is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self.cache),
            "entries": list(self.cache.keys())[:10]  # Sample
        }


if __name__ == "__main__":
    # Test translation
    import os
    from dotenv import load_dotenv
    
    load_dotenv(".env.local")
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        exit(1)
    
    try:
        translator = GeminiTranslator(api_key=api_key)
        
        # Test single translation
        hindi_text = "नमस्ते, आप कैसे हैं?"
        english = translator.translate_text(hindi_text)
        print(f"\n✅ Translation: {hindi_text} → {english}")
        
        # Test conversation translation
        test_segments = [
            {"speaker": "SPEAKER_00", "text": "नमस्ते सर"},
            {"speaker": "SPEAKER_01", "text": "हाँ, बोलिए।"},
        ]
        
        translated = translator.translate_conversation(test_segments)
        print(f"\n✅ Translated {len(translated)} segments")
        for seg in translated:
            print(f"  {seg['speaker']}: {seg['translated_text']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
