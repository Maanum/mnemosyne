#!/usr/bin/env python3
"""
Example script demonstrating the new modular audio processing system.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import setup_logging
from audio import AudioProcessor
from config.settings import path_config


def main():
    """Example usage of the AudioProcessor."""
    
    # Set up logging
    setup_logging(level="INFO")
    
    # Initialize the audio processor
    processor = AudioProcessor()
    
    # Example 1: Process a single file
    print("=== Single File Processing Example ===")
    
    # Check if there are any audio files in the data/audio directory
    audio_dir = path_config.audio_dir
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*"))
        if audio_files:
            # Process the first audio file found
            input_file = audio_files[0]
            print(f"Processing single file: {input_file}")
            
            result = processor.process_single_file(input_file)
            
            if result["success"]:
                print(f"✅ Processing successful!")
                print(f"   Input: {result['input_file']}")
                print(f"   Diarization: {result['diarization_file']}")
                print(f"   Transcription: {result['transcription_file']}")
                print(f"   Cleaned: {result['cleaned_file']}")
                
                # Show statistics if available
                if "diarization_stats" in result:
                    stats = result["diarization_stats"]
                    print(f"   Speakers detected: {stats.get('unique_speakers', 'N/A')}")
                    print(f"   Total segments: {stats.get('total_segments', 'N/A')}")
            else:
                print(f"❌ Processing failed: {result['error']}")
        else:
            print("No audio files found in data/audio directory")
    else:
        print("data/audio directory does not exist")
    
    # Example 2: Batch processing (commented out for safety)
    print("\n=== Batch Processing Example ===")
    print("To process all files in a directory, uncomment the following code:")
    print("""
    # batch_result = processor.process_directory(
    #     input_dir=path_config.audio_dir,
    #     output_dir=path_config.transcripts_dir
    # )
    # 
    # print(f"Batch processing completed:")
    # print(f"   Total files: {batch_result['total_files']}")
    # print(f"   Successful: {batch_result['successful_files']}")
    # print(f"   Failed: {batch_result['failed_files']}")
    """)
    
    # Example 3: Using individual components
    print("\n=== Individual Components Example ===")
    print("You can also use the components individually:")
    print("""
    from audio import AudioDiarizer, AudioTranscriber, TranscriptCleaner
    
    # Use diarizer only
    diarizer = AudioDiarizer()
    diarization_file = diarizer.process_file("input.wav")
    
    # Use transcriber only
    transcriber = AudioTranscriber()
    transcription_file = transcriber.transcribe_file("input.wav", diarization_file)
    
    # Use cleaner only
    cleaner = TranscriptCleaner()
    cleaned_file = cleaner.clean_transcript_file(transcription_file)
    """)


if __name__ == "__main__":
    main()
