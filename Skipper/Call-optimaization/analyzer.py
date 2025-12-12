"""
Call Analysis Module
===================
Gemini-powered analysis for sales call context, summary, and efficiency.
"""

import logging
from typing import List, Dict, Optional
import google.generativeai as genai

from config import TranslationConfig
from validation import ValidationError
from retry_utils import retry_on_exception

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Raised when analysis fails."""
    pass


class CallAnalyzer:
    """Analyze sales calls for context, summary, and efficiency using Gemini."""
    
    def __init__(self, 
                 api_key: str,
                 config: Optional[TranslationConfig] = None,
                 call_context: str = "Company calling customers (Caller=Salesperson, Receiver=Customer)"):
        """
        Initialize call analyzer.
        
        Args:
            api_key: Google Gemini API key
            config: Optional translation configuration
            call_context: Context about the type of calls
            
        Raises:
            ValidationError: If API key is invalid
        """
        if not api_key:
            raise ValidationError("Gemini API key is required")
        
        self.api_key = api_key
        self.config = config or TranslationConfig()
        self.call_context = call_context
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Gemini AI client."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.config.model)
            logger.info(f"✓ Gemini analyzer initialized ({self.config.model})")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise AnalysisError(f"Client initialization failed: {e}") from e
    
    def analyze_call(self, segments: List[Dict]) -> Dict[str, str]:
        """
        Perform complete call analysis.
        
        Args:
            segments: List of conversation segments with 'translated_text' or 'text'
            
        Returns:
            Dictionary with 'context', 'summary', and 'efficiency' keys
        """
        if not segments:
            raise AnalysisError("No segments provided for analysis")
        
        logger.info("Running call analysis...")
        
        # Build conversation text
        conversation = self._build_conversation(segments)
        
        # Run all analyses
        results = {
            'context': self.analyze_context(conversation),
            'summary': self.analyze_summary(conversation),
            'efficiency': self.analyze_efficiency(conversation)
        }
        
        logger.info("✓ Call analysis complete")
        return results
    
    def _build_conversation(self, segments: List[Dict]) -> str:
        """Build formatted conversation from segments."""
        lines = []
        for seg in segments:
            speaker = seg.get('speaker', 'Unknown').replace('SPEAKER_', 'Speaker ')
            # Use translated text if available, otherwise original
            text = seg.get('translated_text') or seg.get('text', '')
            lines.append(f"[{speaker}]: {text}")
        return "\n".join(lines)
    
    @retry_on_exception(
        exceptions=(ConnectionError, TimeoutError),
        max_attempts=2,
        base_delay=3.0
    )
    def analyze_context(self, conversation: str) -> str:
        """
        Extract call context.
        
        Args:
            conversation: Formatted conversation text
            
        Returns:
            Context analysis
        """
        logger.info("  Analyzing context...")
        
        prompt = f"""Analyze this sales call conversation and provide the CONTEXT.

CONTEXT: {self.call_context}

Conversation:
{conversation}

Provide a brief context covering:
1. What product/service is being discussed
2. Purpose of the call (follow-up, feedback, survey, upselling, etc.)
3. Customer profile (if discernible)
4. Any specific customer needs mentioned

Keep it concise (3-5 bullet points).
Format with bullet points using '•' symbol."""
        
        return self._call_gemini(prompt)
    
    @retry_on_exception(
        exceptions=(ConnectionError, TimeoutError),
        max_attempts=2,
        base_delay=3.0
    )
    def analyze_summary(self, conversation: str) -> str:
        """
        Generate call summary.
        
        Args:
            conversation: Formatted conversation text
            
        Returns:
            Call summary
        """
        logger.info("  Analyzing summary...")
        
        prompt = f"""Analyze this sales call conversation and provide a SUMMARY.

CONTEXT: {self.call_context}

Conversation:
{conversation}

Provide a comprehensive summary covering:
1. Key discussion points
2. Customer responses and feedback
3. Information collected
4. Any agreements or commitments made
5. Next steps or follow-up actions
6. Overall outcome of the call

Be concise but comprehensive. Use bullet points for clarity."""
        
        return self._call_gemini(prompt)
    
    @retry_on_exception(
        exceptions=(ConnectionError, TimeoutError),
        max_attempts=2,
        base_delay=3.0
    )
    def analyze_efficiency(self, conversation: str) -> str:
        """
        Analyze salesperson efficiency and performance.
        
        Args:
            conversation: Formatted conversation text
            
        Returns:
            Efficiency analysis
        """
        logger.info("  Analyzing efficiency...")
        
        prompt = f"""Analyze this sales call from a SALESPERSON PERFORMANCE perspective.

CONTEXT: {self.call_context}

Conversation:
{conversation}

Provide a brief efficiency report covering:

## MISTAKES MADE:
- Communication errors or unclear statements
- Missed opportunities to gather information
- Poor handling of customer responses
- Lack of clarity or professionalism
- Interrupting or talking over customer

## WHAT COULD BE DONE BETTER:
- Specific improvements for next call
- Better approaches to use
- Questions that should have been asked
- Information that could have been provided
- Techniques to improve engagement

## STRENGTHS:
- What the salesperson did well
- Effective techniques used

Keep it SHORT and ACTIONABLE. Focus on practical improvements.
Be constructive but honest about issues.
Use bullet points with clear, specific recommendations."""
        
        return self._call_gemini(prompt)
    
    def _call_gemini(self, prompt: str) -> str:
        """
        Make API call to Gemini.
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            Generated text
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,  # Lower temperature for factual analysis
                    "max_output_tokens": self.config.max_tokens
                }
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise AnalysisError(f"Analysis failed: {e}") from e
    
    def analyze_with_script(self,
                           segments: List[Dict],
                           expected_script: Optional[Dict] = None) -> Dict:
        """
        Analyze call against expected script/flow.
        
        Args:
            segments: Conversation segments
            expected_script: Optional dictionary defining expected flow
            
        Returns:
            Analysis with script adherence metrics
        """
        if not expected_script:
            logger.warning("No script provided, running standard analysis")
            return self.analyze_call(segments)
        
        conversation = self._build_conversation(segments)
        
        # Build script adherence prompt
        script_points = "\n".join([f"- {point}" for point in expected_script.get('required_points', [])])
        
        prompt = f"""Analyze this sales call for SCRIPT ADHERENCE.

EXPECTED SCRIPT POINTS:
{script_points}

CONVERSATION:
{conversation}

Analyze:
1. Which script points were covered? (List them)
2. Which script points were MISSED? (List them)
3. Were there any deviations from expected flow?
4. Overall adherence score (0-100%)

Provide specific examples from the conversation."""
        
        adherence_analysis = self._call_gemini(prompt)
        
        # Combine with standard analysis
        standard_analysis = self.analyze_call(segments)
        standard_analysis['script_adherence'] = adherence_analysis
        
        return standard_analysis


if __name__ == "__main__":
    # Test analyzer
    import os
    from dotenv import load_dotenv
    
    load_dotenv(".env.local")
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        exit(1)
    
    # Test with sample segments
    test_segments = [
        {
            "speaker": "SPEAKER_00",
            "translated_text": "Hello Sir, Shilpa spoke on behalf of Keep Sathi. You spoke to the student hall, right?"
        },
        {
            "speaker": "SPEAKER_01",
            "translated_text": "Yes, tell me."
        },
        {
            "speaker": "SPEAKER_00",
            "translated_text": "Sir, do you use our Skipper Sathi app?"
        },
        {
            "speaker": "SPEAKER_01",
            "translated_text": "Yes yes yes."
        },
    ]
    
    try:
        analyzer = CallAnalyzer(api_key=api_key)
        results = analyzer.analyze_call(test_segments)
        
        print("\n" + "="*60)
        print("CALL ANALYSIS RESULTS")
        print("="*60)
        
        for key, value in results.items():
            print(f"\n## {key.upper()}")
            print("-" * 60)
            print(value)
    
    except Exception as e:
        print(f"❌ Error: {e}")
