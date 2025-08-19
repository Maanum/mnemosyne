# Scripts Directory

This directory contains command-line interface scripts for the Mnemosyne project.

## Available Scripts

### 1. `process_audio.py` - Audio Processing Pipeline

Process audio files through the complete transcription pipeline (diarization → transcription → cleaning).

**Usage:**
```bash
# Process a single audio file
python scripts/process_audio.py input.wav

# Process all audio files in a directory
python scripts/process_audio.py --input-dir ./audio --output-dir ./transcripts

# Process with custom settings
python scripts/process_audio.py input.mp4 --output-dir ./output --verbose

# Process OGG files
python scripts/process_audio.py input.ogg --output-dir ./output

# Preview processing without actually running
python scripts/process_audio.py --input-dir ./audio --dry-run
```

**Key Features:**
- Single file or directory processing
- Progress reporting and statistics
- Dry-run mode for preview
- Custom output directories
- Verbose/quiet logging modes

### 2. `populate_database.py` - Database Population

Populate the Weaviate database with transcript data from CSV files.

**Usage:**
```bash
# Populate from default source directory
python scripts/populate_database.py

# Populate from specific directory
python scripts/populate_database.py --input-dir ./my_data

# Reset schema and populate
python scripts/populate_database.py --reset-schema --input-dir ./data

# Preview what would be ingested
python scripts/populate_database.py --dry-run --input-dir ./data

# Validate existing data
python scripts/populate_database.py --validate-only
```

**Key Features:**
- Schema management (create, validate, reset)
- Data validation and reporting
- Batch processing with progress tracking
- Dry-run mode for preview
- Comprehensive statistics

### 3. `start_web.py` - Web Interface Launcher

Launches the modern web interface for querying the transcript database.

**Usage:**
```bash
# Start the web interface
python scripts/start_web.py
```

**Key Features:**
- Modern, Claude-style chat interface
- Real-time responses with typing indicators
- Connection status monitoring
- Professional formatting with metadata
- Responsive design for all devices

## Common Options

All scripts support these common options:

- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress all output except errors
- `--log-file`: Specify custom log file path
- `--help`: Show help message

## Environment Setup

Before running the scripts, ensure:

1. **Dependencies installed:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment variables set:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

3. **Database running:**
   - Weaviate should be running (embedded or remote)
   - Database connection should be healthy

## Example Workflow

Here's a typical workflow using the scripts:

```bash
# 1. Process audio files
python scripts/process_audio.py --input-dir ./raw_audio --output-dir ./transcripts

# 2. Populate database with processed transcripts
python scripts/populate_database.py --input-dir ./transcripts --reset-schema

# 3. Start the web interface
python scripts/start_web.py
```

## Troubleshooting

### Common Issues

1. **Database connection errors:**
   - Ensure Weaviate is running
   - Check environment variables
   - Verify network connectivity

2. **Audio processing errors:**
   - Check if FFmpeg is installed
   - Verify audio file formats are supported
   - Ensure HuggingFace auth token is set

3. **Import errors:**
   - Ensure you're running from the project root
   - Check that all dependencies are installed
   - Verify the src directory structure

### Getting Help

- Use `--help` on any script for detailed usage information
- Check the logs in the `logs/` directory
- Enable verbose mode with `--verbose` for detailed output
- Use debug mode in query interface for troubleshooting

## Script Architecture

All scripts follow a consistent architecture:

1. **Argument parsing** with argparse
2. **Logging setup** with configurable levels
3. **Component initialization** with error handling
4. **Main processing logic** with progress reporting
5. **Clean exit** with appropriate status codes

The scripts use the refactored modular components from the `src/` directory, providing a clean separation between the CLI interface and the core functionality.
