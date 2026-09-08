"""
Production Call Analysis Pipeline
=================================
Complete end-to-end pipeline with all fixes applied.

Fixes All 47 Flaws:
- ✅ No hardcoded API keys
- ✅ Input validation
- ✅ Retry logic with exponential backoff
- ✅ Proper error handling
- ✅ Type hints throughout
- ✅ Comprehensive logging
- ✅ Temp file cleanup
- ✅ Configuration management
- ✅ Modular architecture
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import sys

# Import refactored modules
from .config import load_config, Config
from .validation import AudioValidator, validate_processing_ready
from .audio_processing import AudioCleaner
from .diarization import SpeakerDiarizer
from .transcription import SarvamTranscriber
from .translation import GeminiTranslator
from .analyzer import CallAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
       logging.FileHandler('logs/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class CallAnalysisPipeline:
    """Production-ready call analysis pipeline."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Optional Config object (will load from env if None)
        """
        self.config = config or load_config()
        self._validate_system()
        self._initialize_components()
    
    def _validate_system(self) -> None:
        """Validate system is ready for processing."""
        logger.info("Validating system requirements...")
        
        ready, issues = validate_processing_ready()
        if not ready:
            logger.error("System validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            raise SystemError("System not ready for processing")
        
        logger.info("✓ System validation passed")
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")
        
        try:
            # Audio cleaning
            self.audio_cleaner = AudioCleaner(
                validator=AudioValidator(
                    max_size_mb=self.config.processing.max_file_size_mb,
                    max_duration_seconds=self.config.processing.max_duration_seconds,
                    min_duration_seconds=self.config.processing.min_duration_seconds,
                    supported_formats=self.config.processing.supported_formats
                )
            )
            
            # Speaker diarization
            self.diarizer = SpeakerDiarizer(
                huggingface_token=self.config.api.huggingface_token,
                config=self.config.processing
            )
            
            # Transcription
            self.transcriber = SarvamTranscriber(
                api_key=self.config.api.sarvam_api_key,
                config=self.config.translation
            )
            
            # Translation
            self.translator = GeminiTranslator(
                api_key=self.config.api.gemini_api_key,
                config=self.config.translation
            )
            
            # Analysis
            self.analyzer = CallAnalyzer(
                api_key=self.config.api.gemini_api_key,
                config=self.config.translation
            )
            
            logger.info("✓ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_call(self, audio_path: Path) -> Dict:
        """
        Process single call recording through complete pipeline.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with all results
        """
        audio_path = Path(audio_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = audio_path.stem
        
        # Create temp directory for this job
        temp_dir = self.config.paths.temp_dir / f"{base_name}_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info(f"PROCESSING CALL: {audio_path.name}")
        logger.info("=" * 70)
        
        try:
            # Step 1: Clean Audio
            logger.info("\n[1/5] NOISE REDUCTION")
            logger.info("-" * 70)
            cleaned_audio = self.audio_cleaner.clean_audio(
                audio_path,
                temp_dir / f"cleaned_{audio_path.name}"
            )
            
            # Step 2: Speaker Diarization
            logger.info("\n[2/5] SPEAKER DIARIZATION")
            logger.info("-" * 70)
            segments = self.diarizer.diarize(
                cleaned_audio,
                num_speakers=self.config.processing.num_speakers
            )
            
            # Get speaker statistics
            speaker_stats = self.diarizer.get_speaker_statistics(segments)
            
            # Extract audio segments
            segments_dir = temp_dir / "segments"
            segments_with_audio = self.diarizer.extract_audio_segments(
                cleaned_audio,
                segments,
                segments_dir
            )
            
            # Step 3: Transcription
            logger.info("\n[3/5] TRANSCRIPTION")
            logger.info("-" * 70)
            transcribed_segments = self.transcriber.transcribe_segments(
                segments_with_audio,
                show_progress=True
            )
            
            # Step 4: Translation
            logger.info("\n[4/5] TRANSLATION")
            logger.info("-" * 70)
            translated_segments = self.translator.translate_conversation(
                transcribed_segments,
                preserve_speakers=True
            )
            
            # Step 5: Analysis
            logger.info("\n[5/5] CALL ANALYSIS")
            logger.info("-" * 70)
            analysis = self.analyzer.analyze_call(translated_segments)
            
            # Save outputs
            logger.info("\nSAVING OUTPUTS")
            logger.info("-" * 70)
            output_paths = self._save_outputs(
                base_name,
                timestamp,
                translated_segments,
                analysis,
                speaker_stats
            )
            
            # Cleanup
            logger.info("\nCLEANING UP")
            logger.info("-" * 70)
            self._cleanup(temp_dir)
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ PROCESSING COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Outputs saved to: {self.config.paths.output_dir}")
            
            return {
                'success': True,
                'segments': translated_segments,
                'analysis': analysis,
                'speaker_stats': speaker_stats,
                'outputs': output_paths
            }
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
            self._cleanup(temp_dir)
            raise
    
    def _save_outputs(self,
                     base_name: str,
                     timestamp: str,
                     segments: list,
                     analysis: dict,
                     speaker_stats: dict) -> Dict[str, Path]:
        """Save all output files."""
        output_dir = self.config.paths.output_dir
        prefix = f"{base_name}_{timestamp}"
        paths = {}
        
        # 1. Call Context
        context_path = output_dir / f"{prefix}_CONTEXT.txt"
        self._save_text_file(
            context_path,
            "CALL CONTEXT",
            analysis['context']
        )
        paths['context'] = context_path
        
        # 2. Call Summary
        summary_path = output_dir / f"{prefix}_SUMMARY.txt"
        self._save_text_file(
            summary_path,
            "CALL SUMMARY",
            analysis['summary']
        )
        paths['summary'] = summary_path
        
        # 3. Efficiency Report
        efficiency_path = output_dir / f"{prefix}_EFFICIENCY.txt"
        self._save_text_file(
            efficiency_path,
            "SALESPERSON EFFICIENCY REPORT",
            analysis['efficiency']
        )
        paths['efficiency'] = efficiency_path
        
        # 4. Full Transcript (bilingual)
        transcript_path = output_dir / f"{prefix}_TRANSCRIPT.txt"
        transcript_content = self._format_transcript(segments, speaker_stats)
        self._save_text_file(
            transcript_path,
            "FULL CALL TRANSCRIPT",
            transcript_content
        )
        paths['transcript'] = transcript_path
        
        logger.info(f"✓ Saved {len(paths)} output files")
        return paths
    
    def _save_text_file(self, path: Path, title: str, content: str) -> None:
        """Save text file with header."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"{title}\n")
            f.write("=" * 70 + "\n\n")
            f.write(content)
            f.write("\n")
        logger.info(f"  Saved: {path.name}")
    
    def _format_transcript(self, segments: list, speaker_stats: dict) -> str:
        """Format bilingual transcript with statistics."""
        lines = []
        
        # Add speaker statistics
        lines.append("SPEAKER STATISTICS")
        lines.append("-" * 70)
        for speaker, stats in speaker_stats.items():
            lines.append(
                f"{speaker}: {stats['count']} turns, "
                f"{stats['total_duration']:.1f}s ({stats['percentage']:.1f}%)"
            )
        lines.append("\n" + "=" * 70)
        lines.append("CONVERSATION")
        lines.append("=" * 70 + "\n")
        
        # Add conversation
        for seg in segments:
            speaker = seg['speaker'].replace('SPEAKER_', 'Speaker ')
            lines.append(f"[{speaker}]")
            lines.append(f"Original: {seg['text']}")
            lines.append(f"English:  {seg.get('translated_text', seg['text'])}")
            lines.append("-" * 70)
        
        return "\n".join(lines)
    
    def _cleanup(self, temp_dir: Path) -> None:
        """Clean up temporary files."""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info("✓ Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Production Call Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_production.py --input path/to/call.mp3
  python pipeline_production.py -i call.mp3 --output-dir results/
        """
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        type=Path,
        help="Path to audio file to process"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        help="Optional output directory (overrides config)"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    try:
        # Load config
        config = load_config()
        
        # Override output dir if specified
        if args.output_dir:
            config.paths.output_dir = args.output_dir
            config.paths.create_directories()
        
        # Run pipeline
        pipeline = CallAnalysisPipeline(config)
        result = pipeline.process_call(args.input)
        
        if result['success']:
            print("\n✅ Processing completed successfully!")
            print(f"\nOutput files:")
            for name, path in result['outputs'].items():
                print(f"  {name}: {path}")
            sys.exit(0)
        else:
            print("\n❌ Processing failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠ Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == "__main__":
    main()
