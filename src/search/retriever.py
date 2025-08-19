"""
Transcript retrieval module for vector search operations.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from database.client import get_client
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import path_config

logger = logging.getLogger(__name__)


class TranscriptRetriever:
    """
    Handles vector search operations for transcript retrieval.
    
    This class manages the process of searching through transcript data
    using vector similarity, with support for filtering, ranking, and
    context length management.
    """
    
    def __init__(self, class_name: str = "Transcript"):
        """
        Initialize the TranscriptRetriever.
        
        Args:
            class_name: Name of the Weaviate class to search.
        """
        self.class_name = class_name
        self.client = get_client()
        logger.info(f"TranscriptRetriever initialized for class: {class_name}")
    
    def search_transcripts(
        self,
        query: str,
        limit: Optional[int] = None,
        excluded_speakers: Optional[List[str]] = None,
        min_similarity: Optional[float] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant transcripts using vector similarity.
        
        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            excluded_speakers: List of speaker names to exclude from results.
            min_similarity: Minimum similarity score threshold.
            include_metadata: Whether to include metadata in results.
            
        Returns:
            List of transcript dictionaries with search results.
        """
        try:
            logger.info(f"Searching transcripts for query: '{query[:50]}...'")
            
            if not self.client.is_healthy():
                logger.error("Database connection is unhealthy")
                return []
            
            # Prepare query fields
            fields = ["text", "speaker", "timestamp"]
            if include_metadata:
                fields.extend(["_additional {distance}"])
            
            # Build query
            with self.client.get_connection() as client:
                collection = client.collections.get(self.class_name)
                response = collection.query.near_text(
                    query=query,
                    limit=limit or 10,
                    return_properties=["text", "speaker", "timestamp"]
                )
                
                if not response.objects:
                    logger.warning("No data returned from search")
                    return []
                
                transcripts = []
                for obj in response.objects:
                    transcript = {
                        "text": obj.properties.get("text", ""),
                        "speaker": obj.properties.get("speaker", ""),
                        "timestamp": obj.properties.get("timestamp", ""),
                        "similarity": 1.0 - obj.metadata.distance if obj.metadata.distance else 0.0
                    }
                    transcripts.append(transcript)
                logger.info(f"Retrieved {len(transcripts)} transcripts from search")
                
                # Apply filters and ranking
                filtered_transcripts = self._filter_and_rank_results(
                    transcripts, excluded_speakers, min_similarity
                )
                
                logger.info(f"Returning {len(filtered_transcripts)} filtered transcripts")
                return filtered_transcripts
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _filter_and_rank_results(
        self,
        transcripts: List[Dict[str, Any]],
        excluded_speakers: Optional[List[str]] = None,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter and rank search results.
        
        Args:
            transcripts: Raw search results.
            excluded_speakers: List of speaker names to exclude.
            min_similarity: Minimum similarity score threshold.
            
        Returns:
            Filtered and ranked transcript list.
        """
        try:
            filtered_transcripts = []
            
            for transcript in transcripts:
                # Apply speaker filter
                if excluded_speakers:
                    speaker = transcript.get("speaker", "")
                    if speaker in excluded_speakers:
                        continue
                
                # Apply similarity filter
                if min_similarity is not None:
                    distance = transcript.get("_additional", {}).get("distance", 1.0)
                    similarity = 1.0 - distance  # Convert distance to similarity
                    if similarity < min_similarity:
                        continue
                
                # Add similarity score if available
                if "_additional" in transcript:
                    distance = transcript["_additional"].get("distance", 1.0)
                    transcript["similarity"] = 1.0 - distance
                
                filtered_transcripts.append(transcript)
            
            # Sort by similarity (highest first)
            filtered_transcripts.sort(
                key=lambda x: x.get("similarity", 0.0), reverse=True
            )
            
            logger.debug(f"Filtered {len(transcripts)} -> {len(filtered_transcripts)} results")
            return filtered_transcripts
            
        except Exception as e:
            logger.error(f"Filtering and ranking failed: {e}")
            return transcripts
    
    def format_context(
        self,
        transcripts: List[Dict[str, Any]],
        max_length: Optional[int] = None,
        include_similarity: bool = False
    ) -> str:
        """
        Format transcripts into context string for LLM.
        
        Args:
            transcripts: List of transcript dictionaries.
            max_length: Maximum context length in characters.
            include_similarity: Whether to include similarity scores.
            
        Returns:
            Formatted context string.
        """
        try:
            logger.debug(f"Formatting {len(transcripts)} transcripts into context")
            
            context_parts = []
            for transcript in transcripts:
                speaker = transcript.get("speaker", "Unknown")
                timestamp = transcript.get("timestamp", "Unknown Time")
                text = transcript.get("text", "No text available")
                
                # Build context part
                context_part = f"{text} ({speaker}, {timestamp})"
                
                # Add similarity if requested
                if include_similarity and "similarity" in transcript:
                    similarity = transcript["similarity"]
                    context_part += f" [similarity: {similarity:.3f}]"
                
                context_parts.append(context_part)
            
            context = " ".join(context_parts)
            
            # Truncate if needed
            if max_length and len(context) > max_length:
                logger.debug(f"Truncating context from {len(context)} to {max_length} characters")
                context = context[:max_length].rsplit(' ', 1)[0] + "..."
            
            logger.debug(f"Formatted context: {len(context)} characters")
            return context
            
        except Exception as e:
            logger.error(f"Context formatting failed: {e}")
            return ""
    
    def truncate_context_smart(
        self,
        context: str,
        max_tokens: int = 3000,
        preserve_speakers: bool = True
    ) -> str:
        """
        Smart context truncation that preserves speaker boundaries.
        
        Args:
            context: Context string to truncate.
            max_tokens: Maximum number of tokens.
            preserve_speakers: Whether to preserve complete speaker segments.
            
        Returns:
            Truncated context string.
        """
        try:
            logger.debug(f"Smart truncating context to {max_tokens} tokens")
            
            # Simple tokenization (words)
            tokens = context.split()
            
            if len(tokens) <= max_tokens:
                return context
            
            # Truncate to max_tokens
            truncated_tokens = tokens[:max_tokens]
            truncated_context = " ".join(truncated_tokens)
            
            if preserve_speakers:
                # Try to find a natural break point (speaker boundary)
                # Look for the last occurrence of a speaker pattern like "(Speaker,"
                import re
                speaker_pattern = r'\([^,]+,\s*\d{2}:\d{2}:\d{2}\)'
                matches = list(re.finditer(speaker_pattern, truncated_context))
                
                if matches:
                    # Find the last complete speaker segment
                    last_match = matches[-1]
                    # Find the next speaker pattern or end of string
                    next_match = re.search(speaker_pattern, truncated_context[last_match.end():])
                    
                    if next_match:
                        # Cut at the start of the next speaker
                        cut_point = last_match.end() + next_match.start()
                        truncated_context = truncated_context[:cut_point]
                    else:
                        # No next speaker, keep everything up to last match
                        truncated_context = truncated_context[:last_match.end()]
                
                # Add ellipsis if truncated
                if len(truncated_context) < len(context):
                    truncated_context += "..."
            
            logger.debug(f"Smart truncation: {len(tokens)} -> {len(truncated_context.split())} tokens")
            return truncated_context
            
        except Exception as e:
            logger.error(f"Smart context truncation failed: {e}")
            # Fallback to simple truncation
            tokens = context.split()
            return " ".join(tokens[:max_tokens])
    
    def get_search_stats(self, transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about search results.
        
        Args:
            transcripts: List of transcript search results.
            
        Returns:
            Dictionary containing search statistics.
        """
        try:
            if not transcripts:
                return {
                    "total_results": 0,
                    "speakers": [],
                    "avg_similarity": 0.0,
                    "total_text_length": 0
                }
            
            # Extract statistics
            speakers = list(set(t.get("speaker", "Unknown") for t in transcripts))
            similarities = [t.get("similarity", 0.0) for t in transcripts if "similarity" in t]
            text_lengths = [len(t.get("text", "")) for t in transcripts]
            
            stats = {
                "total_results": len(transcripts),
                "speakers": speakers,
                "unique_speakers": len(speakers),
                "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
                "min_similarity": min(similarities) if similarities else 0.0,
                "max_similarity": max(similarities) if similarities else 0.0,
                "total_text_length": sum(text_lengths),
                "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0
            }
            
            logger.debug(f"Search stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {"error": str(e)}
