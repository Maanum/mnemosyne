# Mnemosyne

**AI-Powered Knowledge Management for User Research**

Mnemosyne transforms user interview recordings into an instantly searchable knowledge base.

## The Problem

User research teams conduct dozens of interviews and expert network calls but struggle to leverage these insights effectively:

- **Knowledge Silos**: Valuable insights get trapped in video files and scattered notes
- **Memory Gaps**: Relevant customer feedback is forgotten or overlooked during product planning
- **Research Waste**: Teams spend hours manually searching through transcripts or rely on incomplete memory
- **Synthesis Overhead**: Creating research-backed product documentation is time-intensive and often incomplete

## The Solution

Mnemosyne provides an end-to-end RAG (Retrieval-Augmented Generation) system that makes your interview knowledge instantly accessible:

### **Automated Processing Pipeline**
- **Audio Extraction**: Converts video recordings to optimized audio format
- **Speaker Diarization**: Identifies different speakers and their time segments using AI
- **Speech Transcription**: Converts audio to text with speaker attribution and timestamps
- **Intelligent Cleaning**: Consolidates fragmented speech into readable conversations

### **Semantic Search & Retrieval**
- **Vector Database**: Stores transcripts with AI-generated embeddings for semantic search
- **Natural Language Queries**: Ask questions like "What did customers say about our onboarding flow?"
- **Contextual Results**: Returns relevant excerpts with speaker attribution and timestamps
- **Citation Support**: Easy links back to original video segments for verification

### **AI-Powered Insights**
- **Research Assistant**: Uses GPT-4 to synthesize findings across multiple interviews
- **Consensus Detection**: Identifies where interviewees agree or disagree on topics
- **Source Attribution**: All responses include proper citations and timestamps

## Impact

- **Instant Knowledge Access**: Find relevant customer insights in seconds, not hours
- **Better Product Decisions**: Research-backed documentation with proper citations
- **Reduced Synthesis Time**: Automated research compilation and organization
- **Team Alignment**: Shared, searchable knowledge base accessible to all team members

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Maanum/mnemosyne.git
   cd mnemosyne
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   The project will automatically create the necessary directory structure:
   ```
   mnemosyne/
   ├── data/
   │   ├── audio/          # Place your audio/video files here
   │   ├── transcripts/    # Generated transcripts
   │   ├── source_data/    # CSV files for database ingestion
   │   └── weaviate_data/  # Database files
   ├── logs/               # Log files
   └── config/             # Configuration files
   ```
   
   Edit `.env` and add your API keys and configuration:
   ```
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_ORGANIZATION=your_organization_id_here
   OPENAI_PROJECT=your_project_id_here
   
   # Weaviate Configuration
   WEAVIATE_ENABLE_MODULES=text2vec-openai
   WEAVIATE_EMBEDDED=true
   
   # Audio Processing Configuration
   AUDIO_SAMPLE_RATE=16000
   AUDIO_CHANNELS=1
   AUDIO_SAMPLE_WIDTH=2
   WHISPER_MODEL=medium
   
   # Processing Configuration
   MAX_SEGMENTS=  # Leave empty to process all segments
   
   # HuggingFace Token (for speaker diarization)
   PYANNOTE_AUTH_TOKEN=your_huggingface_token_here
   ```

4. **Install system dependencies**
   
   For audio processing, you'll need ffmpeg:
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

   **Note**: FFmpeg is only required for video files (`.mp4`, `.mov`, `.avi`, `.mkv`). Audio files (`.wav`, `.mp3`, `.m4a`, `.ogg`) can be processed directly.

5. **Get HuggingFace token (for speaker diarization)**
   
   The speaker diarization feature requires a HuggingFace token:
   1. Go to [HuggingFace](https://huggingface.co/settings/tokens)
   2. Create a new token
   3. Accept the terms for [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
   4. Add the token to your `.env` file as `PYANNOTE_AUTH_TOKEN=your_token_here`

## Quick Start

The Mnemosyne workflow consists of three simple steps, with each script using sensible defaults so you can run them without any arguments:

### 1. Process Audio Files

Place your audio/video files in the `data/audio/` directory, then run:

```bash
# Simple usage - process all files in default directory (data/audio/)
python scripts/process_audio.py

# Or process a single file
python scripts/process_audio.py path/to/your/audio_file.mp4

# Custom directories
python scripts/process_audio.py --input-dir /path/to/audio --output-dir /path/to/transcripts
```

**Supported formats**: `.mp4`, `.wav`, `.m4a`, `.mp3`, `.mov`, `.ogg`, `.avi`, `.mkv`

**Default behavior**:
- **Input**: `data/audio/` directory
- **Output**: `data/transcripts/` directory  
- **Processing**: All supported audio/video files found

### 2. Manual Review (Optional but Recommended)

**⚠️ Important**: While this step is marked as optional, it's highly recommended for best results. The manual review allows you to:
- Fix transcription errors and typos
- Add context and metadata
- Remove irrelevant content
- Ensure speaker names are consistent
- Add notes or tags for better searchability

**Process:**
1. Import the generated transcript files into Google Sheets
2. Review and clean up transcription errors
3. Add any additional metadata or context
4. Export as CSV files to `data/source_data/`

**📋 Required CSV Format**

Your CSV files must have these exact column headers:

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| `Text` | ✅ Yes | The transcribed text content | "I think the onboarding flow is too complicated" |
| `Speaker` | ✅ Yes | Name of the person speaking | "John Doe" or "Interviewer" |
| `Timestamp` | ✅ Yes | Time marker in the recording | "00:15:30" or "15:30" |

**Example CSV content:**
```csv
Text,Speaker,Timestamp
"Hi, thanks for joining us today.","Interviewer","00:00:00"
"I'm excited to share my thoughts about the product.","John Doe","00:00:05"
"The onboarding process was confusing at first.","John Doe","00:01:20"
"Can you tell us more about that?","Interviewer","00:01:45"
```

**📝 CSV Creation Process:**

1. **From cleaned transcripts**: The `process_audio.py` script creates cleaned transcript files in `data/transcripts/` with format:
   ```
   Speaker | Timestamp | Text
   ```

2. **Manual conversion**: Import these files into Google Sheets/Excel and:
   - Split the "Speaker | Timestamp | Text" format into separate columns
   - Ensure column headers are exactly: `Text`, `Speaker`, `Timestamp`
   - Export as CSV files to `data/source_data/`

3. **Validation**: The system will validate your CSV files and report any issues with missing or malformed data.

**🔄 Data Flow:**
```
Audio/Video Files → process_audio.py → Cleaned Transcripts → Manual Review → CSV Files → populate_database.py → Vector Database → Web Interface
```

**📁 File Formats:**
- **Input**: `.mp4`, `.wav`, `.m4a`, `.mp3`, `.mov`, `.ogg`, `.avi`, `.mkv`
- **Intermediate**: `.txt` (cleaned transcripts with "Speaker | Timestamp | Text" format)
- **Database Input**: `.csv` (with `Text`, `Speaker`, `Timestamp` columns)
- **Output**: Vector embeddings in Weaviate database → Web Interface for querying

### 3. Populate Database

Ingest your CSV files into the vector database:

```bash
# Simple usage - ingest all CSV files from default directory (data/source_data/)
python scripts/populate_database.py

# Reset schema and populate (recommended for first run)
python scripts/populate_database.py --reset-schema

# Custom directory
python scripts/populate_database.py --input-dir /path/to/csv_files
```

**Default behavior**:
- **Input**: `data/source_data/` directory
- **Processing**: All CSV files found
- **Batch size**: 100 records at a time

### 4. Query Your Knowledge Base

Start the web interface:

```bash
python scripts/start_web.py
```

Then open your browser to: **http://localhost:5000**

Ask natural language questions in the modern chat interface:
- "What concerns did users express about the signup process?"
- "How do customers currently solve this problem?" 
- "What features were most requested across all interviews?"

The interface provides a clean, Claude-style experience with:
- ✅ **Real-time responses** with typing indicators
- ✅ **Conversation history** in your browser session
- ✅ **Professional formatting** of AI responses with metadata
- ✅ **Connection monitoring** showing database status and record count
- ✅ **Responsive design** that works on desktop and mobile
- ✅ **Keyboard shortcuts** (Enter to send, Shift+Enter for new line)



## Technology Stack

- **Audio Processing**: 
  - FFmpeg (video extraction)
  - OpenAI Whisper (speech-to-text)
  - pyannote.audio (speaker diarization)
  - librosa + soundfile (audio manipulation)
- **Vector Database**: Weaviate with OpenAI embeddings
- **AI Generation**: GPT-4 for response synthesis
- **Web Interface**: Flask with modern, Claude-style chat UI
- **Language**: Python 3.12+ with modern patterns
- **Data Processing**: pandas, numpy for data manipulation

## Usage Examples

### Process Audio Files
```bash
# Simple usage (uses defaults: data/audio/ → data/transcripts/)
python scripts/process_audio.py

# Process a single file with default output directory
python scripts/process_audio.py interview_01.mp4

# Custom input and output directories
python scripts/process_audio.py --input-dir /path/to/audio --output-dir /path/to/transcripts

# Preview what would be processed (dry run)
python scripts/process_audio.py --dry-run

# Process with verbose output for debugging
python scripts/process_audio.py --verbose

# Process specific file patterns
python scripts/process_audio.py --pattern "*.mp3"
python scripts/process_audio.py --pattern "*.ogg" --verbose
```

### Populate Database
```bash
# Simple usage (uses default: data/source_data/)
python scripts/populate_database.py

# Reset schema and populate (recommended for first run)
python scripts/populate_database.py --reset-schema

# Custom directory and validation
python scripts/populate_database.py --input-dir /path/to/csvs --validate-data

# Preview what would be ingested
python scripts/populate_database.py --dry-run

# Custom batch size for performance tuning
python scripts/populate_database.py --batch-size 50

# Validate existing data without ingestion
python scripts/populate_database.py --validate-only
```



### Web Interface
```bash
# Start the modern web interface
python scripts/start_web.py

# The web interface automatically:
# - Checks database health and connection status
# - Shows number of available transcript segments
# - Provides Claude-style chat experience
# - Includes typing indicators and response metadata
# - Maintains conversation history during session
```

### Advanced Workflow Examples
```bash
# Complete workflow with web interface (recommended)
python scripts/process_audio.py                    # Process audio files
python scripts/populate_database.py --reset-schema # Setup and populate database  
python scripts/start_web.py                        # Start web interface

# Production workflow with validation
python scripts/process_audio.py --verbose          # Process with detailed logging
python scripts/populate_database.py --validate-data --reset-schema  # Validate and populate
python scripts/start_web.py                        # Use web interface for queries

# Custom directories workflow
python scripts/process_audio.py --input-dir /recordings --output-dir /transcripts
python scripts/populate_database.py --input-dir /reviewed_csvs --batch-size 200
```

## Troubleshooting

### Common Issues

1. **Import errors or missing modules:**
   - Ensure you're running from the project root directory
   - Install dependencies: `pip install -r requirements.txt`
   - Check that the `src/` directory structure is correct

2. **Database connection errors:**
   - Verify Weaviate is running (embedded mode is enabled by default)
   - Check your environment variables in `.env`
   - Ensure `WEAVIATE_EMBEDDED=true` is set

3. **Audio processing errors:**
   - Install FFmpeg: `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Ubuntu)
   - Get a HuggingFace token and accept the pyannote/speaker-diarization terms
   - Check that your audio files are in supported formats

4. **OpenAI API errors:**
   - Verify your `OPENAI_API_KEY` is set correctly in `.env`
   - Check your OpenAI account has sufficient credits
   - Ensure the API key has access to the required models

5. **CSV format errors:**
   - **Missing columns**: Ensure your CSV has exactly `Text`, `Speaker`, `Timestamp` headers
   - **Empty fields**: All three columns must have values (no empty cells)
   - **Wrong format**: Text should be in quotes, timestamps can be any format
   - **Encoding issues**: Save CSV files as UTF-8 encoding
   - **Validation errors**: Use `--validate-data` flag to check CSV format before ingestion

6. **CSV validation troubleshooting:**
   ```bash
   # Preview CSV files without ingesting
   python scripts/populate_database.py --dry-run
   
   # Validate CSV format only
   python scripts/populate_database.py --validate-only
   
   # Check specific CSV file format
   python scripts/populate_database.py --input-dir /path/to/csvs --validate-data
   ```

### Getting Help

- **Use `--help`** on any script for detailed usage information:
  ```bash
  python scripts/process_audio.py --help
  python scripts/populate_database.py --help
  ```

- **Check logs** in the `logs/` directory for detailed error information

- **Enable verbose mode** with `--verbose` for detailed output:
  ```bash
  python scripts/process_audio.py --verbose
  python scripts/populate_database.py --verbose
  ```



- **Test with dry run** to see what would be processed:
  ```bash
  python scripts/process_audio.py --dry-run
  python scripts/populate_database.py --dry-run
  ```

### Default Directory Structure

The scripts automatically create and use these directories:

```
mnemosyne/
├── data/
│   ├── audio/          # Default input for process_audio.py
│   ├── transcripts/    # Default output for process_audio.py
│   ├── source_data/    # Default input for populate_database.py
│   └── weaviate_data/  # Database storage
├── logs/               # All script logs
└── config/             # Configuration and prompts
```
