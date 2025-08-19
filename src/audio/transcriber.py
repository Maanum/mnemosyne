"""
Audio transcription module using Whisper for speech-to-text conversion.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import whisper
import numpy as np
import librosa
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import audio_config, path_config

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    Handles audio transcription using Whisper.
    
    This class processes audio files to convert speech to text, with support
    for speaker-segmented transcription using diarization data.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the AudioTranscriber.
        
        Args:
            model_name: Name of the Whisper model to use. If None, uses config default.
        """
        self.model_name = model_name or audio_config.whisper_model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info(f"Whisper model '{self.model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{self.model_name}': {e}")
            raise RuntimeError(f"Could not load Whisper model: {e}")
    
    def load_diarization_from_file(self, input_file_path: Path) -> List[Dict[str, Any]]:
        """
        Load diarization data from a JSON file.
        
        Args:
            input_file_path: Path to the diarization JSON file.
            
        Returns:
            List of diarization segments with start, end, and speaker info.
        """
        try:
            logger.debug(f"Loading diarization data from: {input_file_path}")
            with open(input_file_path, "r", encoding="utf-8") as file:
                diarization_data = json.load(file)
            
            logger.debug(f"Loaded {len(diarization_data)} diarization segments")
            return diarization_data
        except FileNotFoundError:
            logger.error(f"Diarization file not found: {input_file_path}")
            raise FileNotFoundError(f"Diarization file not found: {input_file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in diarization file {input_file_path}: {e}")
            raise ValueError(f"Invalid diarization file format: {e}")
        except Exception as e:
            logger.error(f"Failed to load diarization data from {input_file_path}: {e}")
            raise RuntimeError(f"Could not load diarization data: {e}")
    
    def format_timestamp(self, seconds: float) -> str:
        """
        Format seconds into HH:MM:SS timestamp format.
        
        Args:
            seconds: Time in seconds.
            
        Returns:
            Formatted timestamp string.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def prepare_audio_segment(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        """
        Prepare an audio segment for Whisper processing.
        
        Args:
            audio_array: Audio data as numpy array.
            sr: Sample rate of the audio.
            
        Returns:
            Prepared numpy array for Whisper.
        """
        try:
            # Resample to 16kHz if needed
            if sr != audio_config.sample_rate:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=audio_config.sample_rate)
            
            # Ensure mono (librosa.load with mono=True should handle this, but double-check)
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Normalize to float32
            audio_array = audio_array.astype(np.float32)
            
            return audio_array
        except Exception as e:
            logger.error(f"Failed to prepare audio segment: {e}")
            raise RuntimeError(f"Audio segment preparation failed: {e}")
    
    def transcribe_audio_segment(self, audio_array: np.ndarray, sr: int) -> str:
        """
        Transcribe a single audio segment using Whisper.
        
        Args:
            audio_array: Audio data as numpy array.
            sr: Sample rate of the audio.
            
        Returns:
            Transcribed text.
        """
        try:
            # Prepare audio for Whisper
            prepared_audio = self.prepare_audio_segment(audio_array, sr)
            
            # Transcribe with Whisper
            result = self.model.transcribe(prepared_audio, language="en")
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Failed to transcribe audio segment: {e}")
            return "[TRANSCRIPTION_ERROR]"
    
    def transcribe_file(
        self,
        audio_file_path: Path,
        diarization_file_path: Path,
        output_file_path: Optional[Path] = None,
        max_segments: Optional[int] = None
    ) -> Path:
        """
        Transcribe an audio file using diarization data for speaker segmentation.
        
        Args:
            audio_file_path: Path to the audio file.
            diarization_file_path: Path to the diarization JSON file.
            output_file_path: Path for the output transcription file. If None, generates automatically.
            max_segments: Maximum number of segments to process. If None, processes all.
            
        Returns:
            Path to the saved transcription file.
        """
        try:
            print(f"📂 Step 3/4: Speech transcription...")
            
            # Generate output path if not provided
            if output_file_path is None:
                output_file_path = path_config.get_output_path(audio_file_path.stem)
            
            # Load diarization data
            diarization_data = self.load_diarization_from_file(diarization_file_path)
            
            # Load audio file with librosa
            audio_array, sr = librosa.load(str(audio_file_path), sr=None, mono=True)
            
            # Determine number of segments to process
            total_segments = len(diarization_data)
            if max_segments is not None:
                total_segments = min(max_segments, total_segments)
            
            print(f"   📊 Processing {total_segments} segments...")
            
            # Ensure output directory exists
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process segments with progress bar
            with open(output_file_path, "w", encoding="utf-8") as output_file, \
                 tqdm(total=total_segments, desc="Transcribing", unit="segment") as pbar:
                
                for segment in diarization_data[:total_segments]:
                    try:
                        start_time = segment["start"]
                        end_time = segment["end"]
                        speaker = segment["speaker"]
                        
                        # Convert time to samples
                        start_sample = int(start_time * sr)
                        end_sample = int(end_time * sr)
                        
                        # Extract audio segment
                        segment_audio = audio_array[start_sample:end_sample]
                        
                        # Transcribe segment
                        transcription = self.transcribe_audio_segment(segment_audio, sr)
                        
                        # Format timestamp
                        start_time = self.format_timestamp(segment["start"])
                        
                        # Write to file
                        output_file.write(f"{speaker} | {start_time} | {transcription}\n")
                        
                        # Update progress bar
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process segment {segment}: {e}")
                        # Continue with next segment
                        pbar.update(1)
            
            logger.info(f"Transcription completed successfully: {output_file_path}")
            return output_file_path
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_file_path}: {e}")
            raise RuntimeError(f"Transcription processing failed: {e}")
    
    def get_transcription_stats(self, output_file_path: Path) -> Dict[str, Any]:
        """
        Get statistics about the transcription results.
        
        Args:
            output_file_path: Path to the transcription file.
            
        Returns:
            Dictionary containing transcription statistics.
        """
        try:
            with open(output_file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            
            stats = {
                "total_lines": len(lines),
                "total_words": sum(len(line.split(" | ")[-1].split()) for line in lines if " | " in line),
                "speakers": set(),
                "file_size_mb": output_file_path.stat().st_size / (1024 * 1024)
            }
            
            # Count unique speakers
            for line in lines:
                if " | " in line:
                    speaker = line.split(" | ")[0]
                    stats["speakers"].add(speaker)
            
            stats["unique_speakers"] = len(stats["speakers"])
            stats["speakers"] = list(stats["speakers"])
            
            logger.debug(f"Transcription stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get transcription stats: {e}")
            return {"error": str(e)}
