"""
Audio diarization module using pyannote.audio for speaker identification.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from pyannote.audio import Pipeline
import librosa

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import path_config

logger = logging.getLogger(__name__)


class AudioDiarizer:
    """
    Handles speaker diarization using pyannote.audio.
    
    This class processes audio files to identify and segment different speakers
    in the audio, returning timestamps and speaker labels.
    """
    
    def __init__(self, auth_token: Optional[str] = None, model_name: str = "pyannote/speaker-diarization-3.1"):
        """
        Initialize the AudioDiarizer.
        
        Args:
            auth_token: HuggingFace auth token for pyannote models. If None, uses empty string.
            model_name: Name of the pyannote diarization model to use.
        """
        self.auth_token = auth_token or ""
        self.model_name = model_name
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the pyannote diarization pipeline."""
        try:
            logger.info(f"Initializing pyannote pipeline with model: {self.model_name}")
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.auth_token,
            )
            logger.info("Pyannote pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pyannote pipeline: {e}")
            raise RuntimeError(f"Could not initialize diarization pipeline: {e}")
    
    def get_audio_duration(self, file_path: Path) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            Duration of the audio file in seconds.
        """
        try:
            logger.debug(f"Getting audio duration for: {file_path}")
            duration = librosa.get_duration(path=str(file_path))
            logger.debug(f"Audio duration: {duration:.2f} seconds")
            return duration
        except Exception as e:
            logger.error(f"Failed to get audio duration for {file_path}: {e}")
            raise RuntimeError(f"Could not determine audio duration: {e}")
    
    def diarize_speakers(self, file_path: Path) -> Any:
        """
        Perform speaker diarization on an audio file.
        
        Args:
            file_path: Path to the audio file to process.
            
        Returns:
            Pyannote diarization result object.
        """
        try:
            print(f"   🎯 Analyzing audio segments...")
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            diarization = self.pipeline(str(file_path))
            
            # Count speakers and segments for feedback
            segments = list(diarization.itertracks(yield_label=True))
            speakers = set(speaker for _, _, speaker in segments)
            print(f"   ✅ Found {len(speakers)} speakers in {len(segments)} segments")
            
            return diarization
        except Exception as e:
            logger.error(f"Failed to perform speaker diarization for {file_path}: {e}")
            raise RuntimeError(f"Speaker diarization failed: {e}")
    
    def save_diarization_to_file(self, diarization: Any, output_file_path: Path) -> None:
        """
        Save diarization results to a JSON file.
        
        Args:
            diarization: Pyannote diarization result object.
            output_file_path: Path where to save the diarization data.
        """
        try:
            # Convert diarization to list of dictionaries
            diarization_data = [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker
                }
                for segment, _, speaker in diarization.itertracks(yield_label=True)
            ]
            
            # Ensure output directory exists
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to JSON file
            with open(output_file_path, "w", encoding="utf-8") as file:
                json.dump(diarization_data, file, indent=4)
            
        except Exception as e:
            logger.error(f"Failed to save diarization to {output_file_path}: {e}")
            raise RuntimeError(f"Could not save diarization results: {e}")
    
    def process_file(self, input_file_path: Path, output_file_path: Optional[Path] = None) -> tuple[Path, Any]:
        """
        Process a single audio file through the complete diarization pipeline.
        
        Args:
            input_file_path: Path to the input audio file.
            output_file_path: Path for the output JSON file. If None, generates automatically.
            
        Returns:
            Tuple of (Path to the saved diarization file, diarization object).
        """
        try:
            print(f"📂 Step 2/4: Speaker diarization...")
            
            # Generate output path if not provided
            if output_file_path is None:
                output_file_path = path_config.get_diarization_path(input_file_path.stem)
            
            # Get audio duration for logging
            duration = self.get_audio_duration(input_file_path)
            print(f"   📊 Audio duration: {duration:.1f} seconds")
            
            # Perform diarization
            diarization = self.diarize_speakers(input_file_path)
            
            # Save results
            self.save_diarization_to_file(diarization, output_file_path)
            
            return output_file_path, diarization
            
        except Exception as e:
            logger.error(f"Diarization pipeline failed for {input_file_path}: {e}")
            raise RuntimeError(f"Diarization processing failed: {e}")
    
    def get_diarization_summary(self, diarization: Any) -> Dict[str, Any]:
        """
        Get a summary of diarization results.
        
        Args:
            diarization: Pyannote diarization result object.
            
        Returns:
            Dictionary containing summary statistics.
        """
        try:
            segments = list(diarization.itertracks(yield_label=True))
            speakers = set(speaker for _, _, speaker in segments)
            
            summary = {
                "total_segments": len(segments),
                "unique_speakers": len(speakers),
                "speakers": list(speakers),
                "total_duration": sum(segment.end - segment.start for segment, _, _ in segments)
            }
            
            logger.debug(f"Diarization summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate diarization summary: {e}")
            return {"error": str(e)}
