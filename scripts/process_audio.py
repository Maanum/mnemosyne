#!/usr/bin/env python3
"""
Command-line interface for audio processing pipeline.

This script provides a clean CLI for processing audio files through the complete
pipeline: diarization → transcription → cleaning.
"""

# Suppress warnings before importing libraries
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")
warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")

# Set environment variables to reduce noise
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio import AudioProcessor
from utils import setup_logging
from config.settings import path_config, audio_config


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process audio files through the complete transcription pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in default directory (data/audio/)
  python scripts/process_audio.py

  # Process a single audio file
  python scripts/process_audio.py input.wav

  # Process all audio files in a directory
  python scripts/process_audio.py --input-dir ./audio --output-dir ./transcripts

  # Process with custom settings
  python scripts/process_audio.py input.mp4 --output-dir ./output --verbose

  # Process directory with specific file pattern
  python scripts/process_audio.py --input-dir ./audio --pattern "*.mp3" --batch-size 5

  # Preview processing without actually running
  python scripts/process_audio.py --input-dir ./audio --dry-run
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "input_file",
        nargs="?",
        type=str,
        help="Path to input audio file (default: use data/audio/ directory)"
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        default="data/audio/",
        help="Directory containing audio files to process (default: data/audio/)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/transcripts/",
        help="Output directory for processed files (default: data/transcripts/)"
    )
    
    # Processing options
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="File pattern for directory processing (default: all files)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of files to process in parallel (default: 1)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after processing"
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        help="HuggingFace auth token for pyannote models"
    )
    
    # Control options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be processed without actually running"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: logs/process_audio.log)"
    )
    
    return parser


def validate_input(input_path: str, is_directory: bool = False) -> bool:
    """Validate input path."""
    path = Path(input_path)
    
    if not path.exists():
        print(f"❌ Error: Path does not exist: {input_path}")
        return False
    
    if is_directory and not path.is_dir():
        print(f"❌ Error: Path is not a directory: {input_path}")
        return False
    
    if not is_directory and not path.is_file():
        print(f"❌ Error: Path is not a file: {input_path}")
        return False
    
    return True


def get_supported_files(directory: Path, pattern: str) -> list:
    """Get list of supported audio files in directory."""
    from audio import get_supported_audio_formats
    
    supported_formats = get_supported_audio_formats()
    files = []
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if any(file_path.suffix.lower() == ext for ext in supported_formats):
                files.append(file_path)
    
    return sorted(files)


def check_directory_for_files(directory: Path, pattern: str = "*") -> tuple[list, list]:
    """Check directory for supported and unsupported files."""
    from audio import get_supported_audio_formats
    
    supported_formats = get_supported_audio_formats()
    supported_files = []
    unsupported_files = []
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if any(file_path.suffix.lower() == ext for ext in supported_formats):
                supported_files.append(file_path)
            else:
                unsupported_files.append(file_path)
    
    return sorted(supported_files), sorted(unsupported_files)


def process_single_file(
    input_file: str,
    output_dir: Optional[str],
    processor: AudioProcessor,
    force: bool = False,
    keep_temp: bool = False
) -> bool:
    """Process a single audio file."""
    input_path = Path(input_file)
    
    print(f"🎵 Processing: {input_path.name}")
    
    try:
        result = processor.process_single_file(
            input_file_path=input_path,
            output_dir=output_dir,
            clean_temp_files=not keep_temp
        )
        
        if result["success"]:
            print(f"✅ Successfully processed: {input_path.name}")
            print(f"   📁 Output: {result['cleaned_file']}")
            
            # Show statistics if available
            if "diarization_stats" in result:
                stats = result["diarization_stats"]
                print(f"   🎤 Speakers: {stats.get('unique_speakers', 'N/A')}")
                print(f"   📊 Segments: {stats.get('total_segments', 'N/A')}")
            
            return True
        else:
            print(f"❌ Failed to process: {input_path.name}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {input_path.name}: {e}")
        return False


def process_directory(
    input_dir: str,
    output_dir: Optional[str],
    pattern: str,
    processor: AudioProcessor,
    force: bool = False,
    keep_temp: bool = False,
    dry_run: bool = False
) -> bool:
    """Process all audio files in a directory."""
    input_path = Path(input_dir)
    
    print(f"📁 Processing directory: {input_path}")
    print(f"🔍 Pattern: {pattern}")
    
    # Get supported files
    files = get_supported_files(input_path, pattern)
    
    if not files:
        # Check what files are actually in the directory
        supported_files, unsupported_files = check_directory_for_files(input_path, pattern)
        
        if not supported_files and not unsupported_files:
            print(f"❌ No audio files found in {input_dir}")
            print("   Please add .mp4, .wav, .m4a, .mp3, .mov, .ogg, .avi, or .mkv files to the directory")
        elif unsupported_files:
            print(f"❌ No supported audio files found in {input_dir}")
            print("   Found unsupported files:")
            for file in unsupported_files[:5]:  # Show first 5
                print(f"     - {file.name}")
            if len(unsupported_files) > 5:
                print(f"     ... and {len(unsupported_files) - 5} more")
            print("   Supported formats: .mp4, .wav, .m4a, .mp3, .mov, .ogg, .avi, .mkv")
        return False
    
    print(f"📋 Found {len(files)} files to process:")
    for i, file_path in enumerate(files, 1):
        print(f"   {i}. {file_path.name}")
    
    if dry_run:
        print("🔍 Dry run mode - no files will be processed")
        return True
    
    # Process files
    successful = 0
    failed = 0
    failed_files = []
    
    for i, file_path in enumerate(files, 1):
        try:
            result = processor.process_single_file(
                input_file_path=file_path,
                output_dir=output_dir,
                clean_temp_files=not keep_temp
            )
            
            if result["success"]:
                successful += 1
            else:
                failed += 1
                failed_files.append(file_path.name)
                
        except Exception as e:
            print(f"\n❌ ERROR processing {file_path.name}:")
            print(f"   💥 {str(e)}")
            print(f"   ⏭️  Skipping to next file...")
            failed += 1
            failed_files.append(file_path.name)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🏁 BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful}/{len(files)} files")
    if failed_files:
        print(f"❌ Failed files:")
        for failed_file in failed_files:
            print(f"   • {failed_file}")
    
    return failed == 0


def main():
    """Main function."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = "INFO"
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "ERROR"
    
    log_file = args.log_file or path_config.project_root / "logs" / "process_audio.log"
    setup_logging(level=log_level, log_file=log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting audio processing pipeline")
    
    # Print header
    print(f"\n🎯 MNEMOSYNE AUDIO PROCESSOR")
    print(f"{'='*50}")
    
    # Set defaults if no input specified
    if not args.input_file and not args.input_dir:
        # Use the default from argparse
        args.input_dir = "data/audio/"
    
    # Display paths being used
    if args.input_file:
        print(f"📁 Input file: {args.input_file}")
    else:
        print(f"📁 Input directory: {args.input_dir}")
    
    if args.output_dir:
        print(f"📁 Output directory: {args.output_dir}")
    
    # Ensure default directories exist
    if not args.input_file:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"❌ Error: Input directory does not exist: {args.input_dir}")
            print("   Please create the directory or specify a different path")
            sys.exit(1)
    
    # Validate inputs
    if args.input_file:
        if not validate_input(args.input_file, is_directory=False):
            sys.exit(1)
    elif args.input_dir:
        if not validate_input(args.input_dir, is_directory=True):
            sys.exit(1)
    
    # Initialize processor
    try:
        # Use command-line auth token or fall back to config
        auth_token = args.auth_token or audio_config.auth_token
        processor = AudioProcessor(auth_token=auth_token)
        logger.info("AudioProcessor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Process files
    success = False
    
    try:
        if args.input_file:
            # Single file processing
            success = process_single_file(
                input_file=args.input_file,
                output_dir=args.output_dir,
                processor=processor,
                force=args.force,
                keep_temp=args.keep_temp
            )
        else:
            # Directory processing
            success = process_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                pattern=args.pattern,
                processor=processor,
                force=args.force,
                keep_temp=args.keep_temp,
                dry_run=args.dry_run
            )
    
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    
    # Exit with appropriate code
    if success:
        print("\n🎉 Processing completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Processing completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
