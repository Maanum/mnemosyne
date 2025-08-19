"""
Transcript cleaning module for consolidating speaker segments.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TranscriptCleaner:
    """
    Handles cleaning and consolidation of transcript files.
    
    This class processes raw transcription files to consolidate consecutive
    segments from the same speaker into single lines for better readability.
    """
    
    def __init__(self):
        """Initialize the TranscriptCleaner."""
        logger.debug("Initializing TranscriptCleaner")
    
    def parse_transcript_line(self, line: str) -> Optional[Dict[str, str]]:
        """
        Parse a single transcript line into components.
        
        Args:
            line: Raw transcript line in format "speaker | timestamp | text"
            
        Returns:
            Dictionary with speaker, timestamp, and text, or None if invalid format.
        """
        try:
            line = line.strip()
            if not line:
                return None
            
            parts = line.split(" | ")
            if len(parts) < 3:
                logger.warning(f"Invalid transcript line format: {line}")
                return None
            
            speaker = parts[0].strip()
            timestamp = parts[1].strip()
            text = " | ".join(parts[2:]).strip()
            
            return {
                "speaker": speaker,
                "timestamp": timestamp,
                "text": text
            }
        except Exception as e:
            logger.warning(f"Failed to parse transcript line '{line}': {e}")
            return None
    
    def clean_transcript_file(self, input_file_path: Path, output_file_path: Optional[Path] = None) -> Path:
        """
        Clean and consolidate a transcript file.
        
        Args:
            input_file_path: Path to the input transcript file.
            output_file_path: Path for the output cleaned file. If None, generates automatically.
            
        Returns:
            Path to the saved cleaned transcript file.
        """
        try:
            print(f"📂 Step 4/4: Transcript cleaning...")
            
            # Generate output path if not provided
            if output_file_path is None:
                output_file_path = Path(str(input_file_path).replace(".txt", "_cleaned.txt"))
            
            # Ensure output directory exists
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read input file
            with open(input_file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            
            print(f"   📊 Processing {len(lines)} transcript lines...")
            
            # Process lines
            processed_lines = []
            previous_speaker = None
            current_text = ""
            start_timestamp = ""
            
            for line_num, line in enumerate(lines, 1):
                try:
                    parsed = self.parse_transcript_line(line)
                    if parsed is None:
                        continue
                    
                    speaker = parsed["speaker"]
                    timestamp = parsed["timestamp"]
                    text = parsed["text"]
                    
                    if speaker == previous_speaker:
                        # Append text if the same speaker as previous line
                        current_text += " " + text
                        logger.debug(f"Consolidating text for speaker {speaker}")
                    else:
                        # Save the previous speaker's consolidated text
                        if previous_speaker is not None:
                            consolidated_line = f"{previous_speaker} | {start_timestamp} | {current_text}\n"
                            processed_lines.append(consolidated_line)
                            logger.debug(f"Added consolidated line for {previous_speaker}")
                        
                        # Start a new line for a new speaker
                        previous_speaker = speaker
                        start_timestamp = timestamp
                        current_text = text
                        
                except Exception as e:
                    logger.warning(f"Failed to process line {line_num}: {e}")
                    continue
            
            # Don't forget to add the last speaker's consolidated text
            if previous_speaker is not None:
                consolidated_line = f"{previous_speaker} | {start_timestamp} | {current_text}\n"
                processed_lines.append(consolidated_line)
                logger.debug(f"Added final consolidated line for {previous_speaker}")
            
            # Write processed lines to output file
            logger.debug(f"Writing {len(processed_lines)} consolidated lines to: {output_file_path}")
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.writelines(processed_lines)
            
            logger.info(f"Transcript cleaning completed successfully: {output_file_path}")
            return output_file_path
            
        except FileNotFoundError:
            logger.error(f"Input transcript file not found: {input_file_path}")
            raise FileNotFoundError(f"Input transcript file not found: {input_file_path}")
        except Exception as e:
            logger.error(f"Transcript cleaning failed for {input_file_path}: {e}")
            raise RuntimeError(f"Transcript cleaning failed: {e}")
    
    def get_cleaning_stats(self, input_file_path: Path, output_file_path: Path) -> Dict[str, Any]:
        """
        Get statistics about the cleaning process.
        
        Args:
            input_file_path: Path to the original transcript file.
            output_file_path: Path to the cleaned transcript file.
            
        Returns:
            Dictionary containing cleaning statistics.
        """
        try:
            # Read original file
            with open(input_file_path, "r", encoding="utf-8") as file:
                original_lines = file.readlines()
            
            # Read cleaned file
            with open(output_file_path, "r", encoding="utf-8") as file:
                cleaned_lines = file.readlines()
            
            # Calculate statistics
            stats = {
                "original_lines": len(original_lines),
                "cleaned_lines": len(cleaned_lines),
                "lines_consolidated": len(original_lines) - len(cleaned_lines),
                "consolidation_ratio": len(cleaned_lines) / len(original_lines) if original_lines else 0,
                "original_file_size_mb": input_file_path.stat().st_size / (1024 * 1024),
                "cleaned_file_size_mb": output_file_path.stat().st_size / (1024 * 1024)
            }
            
            # Count speakers in both files
            original_speakers = set()
            cleaned_speakers = set()
            
            for line in original_lines:
                parsed = self.parse_transcript_line(line)
                if parsed:
                    original_speakers.add(parsed["speaker"])
            
            for line in cleaned_lines:
                parsed = self.parse_transcript_line(line)
                if parsed:
                    cleaned_speakers.add(parsed["speaker"])
            
            stats["original_speakers"] = len(original_speakers)
            stats["cleaned_speakers"] = len(cleaned_speakers)
            stats["speakers_consolidated"] = len(original_speakers) - len(cleaned_speakers)
            
            logger.debug(f"Cleaning stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cleaning stats: {e}")
            return {"error": str(e)}
    
    def validate_transcript_format(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate the format of a transcript file.
        
        Args:
            file_path: Path to the transcript file to validate.
            
        Returns:
            Dictionary containing validation results.
        """
        try:
            logger.debug(f"Validating transcript format: {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            
            validation_results = {
                "total_lines": len(lines),
                "valid_lines": 0,
                "invalid_lines": 0,
                "empty_lines": 0,
                "format_errors": [],
                "speakers": set(),
                "sample_valid_line": None
            }
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                if not line:
                    validation_results["empty_lines"] += 1
                    continue
                
                parsed = self.parse_transcript_line(line)
                if parsed is None:
                    validation_results["invalid_lines"] += 1
                    validation_results["format_errors"].append(f"Line {line_num}: Invalid format")
                else:
                    validation_results["valid_lines"] += 1
                    validation_results["speakers"].add(parsed["speaker"])
                    
                    # Store first valid line as sample
                    if validation_results["sample_valid_line"] is None:
                        validation_results["sample_valid_line"] = line
            
            validation_results["speakers"] = list(validation_results["speakers"])
            validation_results["validity_ratio"] = validation_results["valid_lines"] / len(lines) if lines else 0
            
            logger.debug(f"Validation results: {validation_results}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate transcript format: {e}")
            return {"error": str(e)}
