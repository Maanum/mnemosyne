"""
Mnemosyne - Audio transcription and search system
"""

__version__ = "0.1.0"
__author__ = "Kristofer"

# Import main modules for easy access
from .audio import AudioProcessor, AudioDiarizer, AudioTranscriber, TranscriptCleaner
from .database import WeaviateClient, get_client, SchemaManager, DataIngester
from .search import TranscriptRetriever, ResponseGenerator, RAGPipeline
from .utils import setup_logging, get_logger

__all__ = [
    "AudioProcessor",
    "AudioDiarizer", 
    "AudioTranscriber",
    "TranscriptCleaner",
    "WeaviateClient",
    "get_client",
    "SchemaManager",
    "DataIngester",
    "TranscriptRetriever",
    "ResponseGenerator",
    "RAGPipeline",
    "setup_logging",
    "get_logger",
]
