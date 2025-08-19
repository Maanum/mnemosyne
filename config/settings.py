# Suppress noisy library warnings immediately
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

"""
Centralized configuration management for Mnemosyne project.
Handles environment variables, file paths, API settings, and auto-directory creation.
"""

import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import weaviate

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def configure_logging():
    """Configure logging levels for external libraries."""
    # Silence noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("weaviate").setLevel(logging.WARNING)
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torchaudio").setLevel(logging.WARNING)
    logging.getLogger("librosa").setLevel(logging.WARNING)
    logging.getLogger("soundfile").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# Set the environment variable that Weaviate expects for OpenAI API key
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_APIKEY"] = os.getenv("OPENAI_API_KEY")

# Set Weaviate logging environment variables
os.environ.setdefault('WEAVIATE_LOG_LEVEL', 'ERROR')
os.environ.setdefault('GOLOG_LOG_LEVEL', 'ERROR') 
os.environ.setdefault('RUST_LOG', 'ERROR')

# Configure logging to suppress noisy libraries immediately
configure_logging()

# Project root directory (parent of config directory)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION", "")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "")

# Weaviate Configuration
WEAVIATE_ENABLE_MODULES = os.getenv("WEAVIATE_ENABLE_MODULES", "text2vec-openai")
WEAVIATE_EMBEDDED = os.getenv("WEAVIATE_EMBEDDED", "true").lower() == "true"

# Audio Processing Configuration
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
AUDIO_SAMPLE_WIDTH = int(os.getenv("AUDIO_SAMPLE_WIDTH", "2"))  # 2 bytes = 16-bit
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
PYANNOTE_AUTH_TOKEN = os.getenv("PYANNOTE_AUTH_TOKEN", "")

# Processing Configuration
MAX_SEGMENTS = os.getenv("MAX_SEGMENTS")
if MAX_SEGMENTS and MAX_SEGMENTS.strip():
    MAX_SEGMENTS = int(MAX_SEGMENTS)
else:
    MAX_SEGMENTS = None

# =============================================================================
# FILE PATHS (relative to project root)
# =============================================================================

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
SOURCE_DATA_DIR = DATA_DIR / "source_data"
WEAVIATE_DATA_DIR = DATA_DIR / "weaviate_data"

# Scripts and configuration
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_DIR = PROJECT_ROOT / "config"
LLM_PROMPT_FILE = CONFIG_DIR / "llm_prompt.json"

# Temporary files
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_AUDIO_FILE = TEMP_DIR / "temp_audio.wav"

# =============================================================================
# AUTO-CREATE DIRECTORIES
# =============================================================================

def ensure_directories_exist():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        AUDIO_DIR,
        TRANSCRIPTS_DIR,
        SOURCE_DATA_DIR,
        WEAVIATE_DATA_DIR,
        SCRIPTS_DIR,
        CONFIG_DIR,
        TEMP_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_openai_config() -> bool:
    """Validate OpenAI configuration."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it in your .env file or environment."
        )
    return True

def validate_weaviate_config() -> bool:
    """Validate Weaviate configuration."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is required for Weaviate text2vec-openai module. "
            "Please set it in your .env file or environment."
        )
    return True

def validate_audio_config() -> bool:
    """Validate audio processing configuration."""
    valid_whisper_models = ["tiny", "base", "small", "medium", "large"]
    if WHISPER_MODEL not in valid_whisper_models:
        raise ValueError(
            f"Invalid WHISPER_MODEL: {WHISPER_MODEL}. "
            f"Must be one of: {valid_whisper_models}"
        )
    return True

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

class OpenAIConfig:
    """OpenAI API configuration."""
    
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.organization = OPENAI_ORGANIZATION
        self.project = OPENAI_PROJECT
        self.model = "gpt-4"  # Default model for chat completions
        validate_openai_config()
    
    def get_client_config(self):
        """Get configuration for OpenAI client."""
        return {
            "api_key": self.api_key,
            "organization": self.organization,
            "project": self.project,
        }

class WeaviateConfig:
    """Weaviate database configuration."""
    
    def __init__(self):
        self.enable_modules = WEAVIATE_ENABLE_MODULES
        self.embedded = WEAVIATE_EMBEDDED
        self.data_dir = WEAVIATE_DATA_DIR
        validate_weaviate_config()
    
    def get_client_config(self):
        """Get configuration for Weaviate client."""
        config = {}
        
        # For v4, we'll set environment variables directly
        if self.embedded:
            import os
            os.environ["ENABLE_MODULES"] = self.enable_modules
            # Redirect Weaviate logs to file instead of console
            os.environ["WEAVIATE_LOG_LEVEL"] = "error"
            os.environ["WEAVIATE_LOG_FORMAT"] = "json"
        
        return config
    
    def get_schema_config(self):
        """Get schema configuration for Transcript class."""
        return {
            "name": "Transcript",
            "vectorizer_config": weaviate.classes.config.Configure.Vectorizer.text2vec_openai(
                model="ada",
                model_version="002",
                type_="text"
            ),
            "properties": [
                weaviate.classes.config.Property(name="text", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="speaker", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="timestamp", data_type=weaviate.classes.config.DataType.TEXT)
            ],
        }

class AudioConfig:
    """Audio processing configuration."""
    
    def __init__(self):
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.channels = AUDIO_CHANNELS
        self.sample_width = AUDIO_SAMPLE_WIDTH
        self.whisper_model = WHISPER_MODEL
        self.max_segments = MAX_SEGMENTS
        self.auth_token = PYANNOTE_AUTH_TOKEN
        validate_audio_config()
    
    def get_audio_settings(self):
        """Get audio processing settings."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "whisper_model": self.whisper_model,
            "max_segments": self.max_segments,
            "auth_token": self.auth_token,
        }

class PathConfig:
    """File path configuration."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = DATA_DIR
        self.audio_dir = AUDIO_DIR
        self.transcripts_dir = TRANSCRIPTS_DIR
        self.source_data_dir = SOURCE_DATA_DIR
        self.weaviate_data_dir = WEAVIATE_DATA_DIR
        self.scripts_dir = SCRIPTS_DIR
        self.config_dir = CONFIG_DIR
        self.temp_dir = TEMP_DIR
        self.temp_audio_file = TEMP_AUDIO_FILE
        self.llm_prompt_file = LLM_PROMPT_FILE
    
    def get_output_path(self, input_filename: str, extension: str = ".txt") -> Path:
        """Generate output path for a given input filename."""
        return self.transcripts_dir / f"{input_filename}{extension}"
    
    def get_diarization_path(self, input_filename: str) -> Path:
        """Generate diarization output path for a given input filename."""
        return self.transcripts_dir / f"{input_filename}.json"
    
    def get_cleaned_path(self, input_filename: str) -> Path:
        """Generate cleaned transcription path for a given input filename."""
        return self.transcripts_dir / f"{input_filename}_cleaned.txt"

# =============================================================================
# GLOBAL CONFIGURATION INSTANCES
# =============================================================================

# Create configuration instances
openai_config = OpenAIConfig()
weaviate_config = WeaviateConfig()
audio_config = AudioConfig()
path_config = PathConfig()

# Ensure all directories exist
ensure_directories_exist()

# Configure logging to suppress noisy libraries
configure_logging()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_supported_audio_formats():
    """Get list of supported audio/video formats."""
    return [".mp4", ".wav", ".m4a", ".mp3", ".mov", ".ogg"]

def is_supported_audio_format(filename: str) -> bool:
    """Check if filename has a supported audio/video format."""
    return any(filename.lower().endswith(ext) for ext in get_supported_audio_formats())

def get_llm_prompt() -> dict:
    """Load LLM prompt from configuration file."""
    try:
        import json
        with open(path_config.llm_prompt_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        # Return default prompt if file doesn't exist
        return {
            "role": "system",
            "content": "You are a research assistant helping the user find relevant passages in a set of transcripts that answer their question. The relevant passages you found will be sent next. When answering the User's questions, you should make references to the specific passages either directly in the text or adding reference in parentheses (such as '(Gilles, 23:11)'. Where important, be sure to indicate cases where all interviewees seem to agree on a topic, and where there are different viewpoints."
        }

# =============================================================================
# CONFIGURATION EXPORTS
# =============================================================================

__all__ = [
    "openai_config",
    "weaviate_config", 
    "audio_config",
    "path_config",
    "get_supported_audio_formats",
    "is_supported_audio_format",
    "get_llm_prompt",
    "ensure_directories_exist",
]
