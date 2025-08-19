"""
Main audio processing orchestrator that coordinates the complete pipeline.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import path_config, audio_config, get_supported_audio_formats, is_supported_audio_format

from .diarizer import AudioDiarizer
from .transcriber import AudioTranscriber
from .cleaner import TranscriptCleaner

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Main orchestrator for the audio processing pipeline.
    
    This class coordinates the complete audio processing workflow:
    1. Audio extraction (if needed)
    2. Speaker diarization
    3. Speech transcription
    4. Transcript cleaning
    
    Supports both single file and batch directory processing.
    """
    
    def __init__(
        self,
        diarizer: Optional[AudioDiarizer] = None,
        transcriber: Optional[AudioTranscriber] = None,
        cleaner: Optional[TranscriptCleaner] = None,
        auth_token: Optional[str] = None
    ):
        """
        Initialize the AudioProcessor with optional dependency injection.
        
        Args:
            diarizer: AudioDiarizer instance. If None, creates a new one.
            transcriber: AudioTranscriber instance. If None, creates a new one.
            cleaner: TranscriptCleaner instance. If None, creates a new one.
            auth_token: HuggingFace auth token for pyannote models.
        """
        logger.info("Initializing AudioProcessor")
        
        # Initialize components with dependency injection
        self.diarizer = diarizer or AudioDiarizer(auth_token=auth_token)
        self.transcriber = transcriber or AudioTranscriber()
        self.cleaner = cleaner or TranscriptCleaner()
        
        # Supported audio formats
        self.supported_formats = get_supported_audio_formats()
        
        logger.info("AudioProcessor initialized successfully")
    
    def extract_audio(self, input_file: Path, output_file: Optional[Path] = None) -> Path:
        """
        Convert input file to WAV format using ffmpeg.
        
        Args:
            input_file: Path to the input audio/video file.
            output_file: Path for the output WAV file. If None, uses temporary file.
            
        Returns:
            Path to the extracted WAV file.
        """
        try:
            logger.info(f"Extracting audio from: {input_file}")
            
            # Generate output path if not provided
            if output_file is None:
                output_file = path_config.temp_audio_file
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", str(input_file),
                "-q:a", "0",
                "-map", "a",
                str(output_file),
                "-y"  # Overwrite output file
            ]
            
            logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True
            )
            
            logger.info(f"Audio extraction completed: {output_file}")
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg extraction failed: {e.stderr.decode()}")
            raise RuntimeError(f"Audio extraction failed: {e.stderr.decode()}")
        except Exception as e:
            logger.error(f"Audio extraction failed for {input_file}: {e}")
            raise RuntimeError(f"Audio extraction failed: {e}")
    
    def process_single_file(
        self,
        input_file_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        clean_temp_files: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single audio file through the complete pipeline.
        
        Args:
            input_file_path: Path to the input audio/video file.
            output_dir: Directory for output files. If None, uses config default.
            clean_temp_files: Whether to clean up temporary files.
            
        Returns:
            Dictionary containing processing results and file paths.
        """
        input_file = Path(input_file_path)
        temp_audio_file = None
        
        try:
            print(f"\n{'='*60}")
            print(f"🎵 PROCESSING: {input_file.name}")
            print(f"{'='*60}")
            
            # Validate input file
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            if not is_supported_audio_format(input_file.name):
                raise ValueError(f"Unsupported file format: {input_file.suffix}")
            
            # Set output directory
            if output_dir is None:
                output_dir = path_config.transcripts_dir
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Extract audio if needed (only for video formats)
            audio_file = input_file
            if input_file.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
                print(f"📂 Step 1/4: Audio extraction...")
                print(f"   🎬 Extracting audio from video format: {input_file.suffix}")
                temp_audio_file = self.extract_audio(input_file)
                audio_file = temp_audio_file
                print(f"   ✅ Audio extracted to temporary file")
            else:
                # For audio formats, librosa can handle them directly
                print(f"📂 Step 1/4: Audio preparation...")
                print(f"   🎵 Using direct audio format: {input_file.suffix}")
            
            # Step 2: Diarization
            diarization_file = output_dir / f"{input_file.stem}.json"
            diarization_file, diarization_result = self.diarizer.process_file(audio_file, diarization_file)
            
            # Step 3: Transcription
            transcription_file = output_dir / f"{input_file.stem}.txt"
            transcription_file = self.transcriber.transcribe_file(
                audio_file, diarization_file, transcription_file
            )
            
            # Step 4: Cleaning
            cleaned_file = output_dir / f"{input_file.stem}_cleaned.txt"
            cleaned_file = self.cleaner.clean_transcript_file(transcription_file, cleaned_file)
            
            # Collect results
            results = {
                "input_file": str(input_file),
                "audio_file": str(audio_file),
                "diarization_file": str(diarization_file),
                "transcription_file": str(transcription_file),
                "cleaned_file": str(cleaned_file),
                "success": True,
                "error": None
            }
            
            # Get processing statistics
            try:
                results["diarization_stats"] = self.diarizer.get_diarization_summary(diarization_result)
                results["transcription_stats"] = self.transcriber.get_transcription_stats(transcription_file)
                results["cleaning_stats"] = self.cleaner.get_cleaning_stats(transcription_file, cleaned_file)
            except Exception as e:
                logger.warning(f"Failed to collect processing stats: {e}")
                results["stats_error"] = str(e)
            
            print(f"\n🎉 COMPLETED: {input_file.name}")
            print(f"   📄 Transcript: {transcription_file.name}")
            print(f"   🧹 Cleaned: {cleaned_file.name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Processing failed for {input_file}: {e}")
            return {
                "input_file": str(input_file),
                "success": False,
                "error": str(e),
                "audio_file": str(temp_audio_file) if temp_audio_file else None,
                "diarization_file": None,
                "transcription_file": None,
                "cleaned_file": None
            }
        
        finally:
            # Clean up temporary files
            if clean_temp_files and temp_audio_file and temp_audio_file.exists():
                try:
                    temp_audio_file.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_audio_file}: {e}")
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        file_pattern: str = "*",
        clean_temp_files: bool = True
    ) -> Dict[str, Any]:
        """
        Process all supported audio files in a directory.
        
        Args:
            input_dir: Directory containing audio files.
            output_dir: Directory for output files. If None, uses config default.
            file_pattern: Glob pattern for file selection.
            clean_temp_files: Whether to clean up temporary files.
            
        Returns:
            Dictionary containing batch processing results.
        """
        input_dir = Path(input_dir)
        
        try:
            logger.info(f"Starting batch processing for directory: {input_dir}")
            
            if not input_dir.exists():
                raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
            # Find all supported audio files
            supported_files = []
            for file_path in input_dir.glob(file_pattern):
                if file_path.is_file() and is_supported_audio_format(file_path.name):
                    supported_files.append(file_path)
            
            logger.info(f"Found {len(supported_files)} supported audio files")
            
            if not supported_files:
                logger.warning("No supported audio files found in directory")
                return {
                    "input_dir": str(input_dir),
                    "total_files": 0,
                    "processed_files": 0,
                    "successful_files": 0,
                    "failed_files": 0,
                    "results": []
                }
            
            # Process each file
            results = []
            successful_count = 0
            failed_count = 0
            
            for file_path in supported_files:
                logger.info(f"Processing file {file_path.name} ({supported_files.index(file_path) + 1}/{len(supported_files)})")
                
                try:
                    result = self.process_single_file(file_path, output_dir, clean_temp_files)
                    results.append(result)
                    
                    if result["success"]:
                        successful_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    results.append({
                        "input_file": str(file_path),
                        "success": False,
                        "error": str(e)
                    })
                    failed_count += 1
            
            batch_results = {
                "input_dir": str(input_dir),
                "total_files": len(supported_files),
                "processed_files": len(results),
                "successful_files": successful_count,
                "failed_files": failed_count,
                "results": results
            }
            
            logger.info(f"Batch processing completed: {successful_count} successful, {failed_count} failed")
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch processing failed for directory {input_dir}: {e}")
            return {
                "input_dir": str(input_dir),
                "success": False,
                "error": str(e),
                "total_files": 0,
                "processed_files": 0,
                "successful_files": 0,
                "failed_files": 0,
                "results": []
            }
    
    def get_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of processing results.
        
        Args:
            results: List of processing results from process_single_file or process_directory.
            
        Returns:
            Dictionary containing processing summary statistics.
        """
        try:
            summary = {
                "total_files": len(results),
                "successful_files": sum(1 for r in results if r.get("success", False)),
                "failed_files": sum(1 for r in results if not r.get("success", False)),
                "success_rate": 0.0,
                "total_processing_time": 0.0,
                "file_types": {},
                "errors": []
            }
            
            if summary["total_files"] > 0:
                summary["success_rate"] = summary["successful_files"] / summary["total_files"]
            
            # Analyze file types and errors
            for result in results:
                if not result.get("success", False):
                    error = result.get("error", "Unknown error")
                    summary["errors"].append(error)
                
                # Count file types
                input_file = Path(result.get("input_file", ""))
                if input_file.suffix:
                    ext = input_file.suffix.lower()
                    summary["file_types"][ext] = summary["file_types"].get(ext, 0) + 1
            
            logger.debug(f"Processing summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate processing summary: {e}")
            return {"error": str(e)}
