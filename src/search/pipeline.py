"""
RAG (Retrieval-Augmented Generation) pipeline module.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .retriever import TranscriptRetriever
from .generator import ResponseGenerator
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import path_config

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.
    
    This class orchestrates the entire RAG workflow including query preprocessing,
    document retrieval, context optimization, response generation, and
    post-processing with debugging capabilities.
    """
    
    def __init__(
        self,
        retriever: Optional[TranscriptRetriever] = None,
        generator: Optional[ResponseGenerator] = None,
        debug_mode: bool = False
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: TranscriptRetriever instance. If None, creates a new one.
            generator: ResponseGenerator instance. If None, creates a new one.
            debug_mode: Whether to enable debug mode with detailed logging.
        """
        self.retriever = retriever or TranscriptRetriever()
        self.generator = generator or ResponseGenerator()
        self.debug_mode = debug_mode
        
        logger.info("RAGPipeline initialized")
        if debug_mode:
            logger.info("Debug mode enabled")
    
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Preprocess the user query.
        
        Args:
            query: Raw user query.
            
        Returns:
            Dictionary containing preprocessed query and metadata.
        """
        try:
            logger.debug(f"Preprocessing query: '{query[:50]}...'")
            
            # Basic preprocessing
            processed_query = query.strip()
            
            # Extract query characteristics
            query_stats = {
                "original_length": len(query),
                "processed_length": len(processed_query),
                "word_count": len(processed_query.split()),
                "has_question_mark": "?" in processed_query,
                "has_quotes": '"' in processed_query or "'" in processed_query
            }
            
            # Detect query type
            query_type = self._detect_query_type(processed_query)
            query_stats["query_type"] = query_type
            
            result = {
                "processed_query": processed_query,
                "stats": query_stats,
                "preprocessing_time": 0.0
            }
            
            logger.debug(f"Query preprocessing completed: {query_stats}")
            return result
            
        except Exception as e:
            logger.error(f"Query preprocessing failed: {e}")
            return {
                "processed_query": query,
                "stats": {"error": str(e)},
                "preprocessing_time": 0.0
            }
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of query.
        
        Args:
            query: Processed query string.
            
        Returns:
            Query type string.
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "when", "where", "who", "why", "how"]):
            return "question"
        elif any(word in query_lower for word in ["find", "search", "look for"]):
            return "search"
        elif any(word in query_lower for word in ["summarize", "summary"]):
            return "summary"
        elif any(word in query_lower for word in ["compare", "difference", "similar"]):
            return "comparison"
        else:
            return "general"
    
    def retrieve_context(
        self,
        query: str,
        limit: Optional[int] = None,
        excluded_speakers: Optional[List[str]] = None,
        min_similarity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for the query.
        
        Args:
            query: User query.
            limit: Maximum number of results to retrieve.
            excluded_speakers: List of speaker names to exclude.
            min_similarity: Minimum similarity score threshold.
            
        Returns:
            Dictionary containing retrieved context and metadata.
        """
        try:
            start_time = time.time()
            logger.info(f"Retrieving context for query: '{query[:50]}...'")
            
            # Retrieve transcripts
            transcripts = self.retriever.search_transcripts(
                query=query,
                limit=limit,
                excluded_speakers=excluded_speakers,
                min_similarity=min_similarity
            )
            
            # Format context
            context = self.retriever.format_context(
                transcripts=transcripts,
                include_similarity=self.debug_mode
            )
            
            # Get retrieval statistics
            retrieval_stats = self.retriever.get_search_stats(transcripts)
            
            result = {
                "transcripts": transcripts,
                "context": context,
                "stats": retrieval_stats,
                "retrieval_time": time.time() - start_time
            }
            
            logger.info(f"Context retrieval completed: {len(transcripts)} transcripts, {len(context)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return {
                "transcripts": [],
                "context": "",
                "stats": {"error": str(e)},
                "retrieval_time": 0.0
            }
    
    def generate_response(
        self,
        query: str,
        context: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response using the retrieved context.
        
        Args:
            query: User query.
            context: Retrieved context.
            temperature: Generation temperature.
            max_tokens: Maximum response tokens.
            
        Returns:
            Dictionary containing generated response and metadata.
        """
        try:
            start_time = time.time()
            logger.info(f"Generating response for query: '{query[:50]}...'")
            
            # Generate response
            generation_result = self.generator.generate_response(
                question=query,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = {
                "response": generation_result.get("response", ""),
                "model": generation_result.get("model", ""),
                "usage": generation_result.get("usage"),
                "generation_time": time.time() - start_time,
                "error": generation_result.get("error")
            }
            
            logger.info(f"Response generation completed: {len(result['response'])} characters")
            return result
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "response": "I'm sorry, I couldn't generate a response at this time.",
                "model": "",
                "usage": None,
                "generation_time": 0.0,
                "error": str(e)
            }
    
    def post_process_response(
        self,
        response: str,
        query: str,
        context: str,
        transcripts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Post-process the generated response.
        
        Args:
            response: Generated response.
            query: Original user query.
            context: Retrieved context.
            transcripts: Retrieved transcripts.
            
        Returns:
            Dictionary containing post-processed response and metadata.
        """
        try:
            logger.debug("Post-processing response")
            
            # Basic post-processing
            processed_response = response.strip()
            
            # Add source information if in debug mode
            if self.debug_mode and transcripts:
                source_info = self._extract_source_info(transcripts)
                processed_response += f"\n\nSources: {source_info}"
            
            # Add confidence indicator
            confidence = self._calculate_confidence(transcripts, query)
            
            result = {
                "processed_response": processed_response,
                "confidence": confidence,
                "source_count": len(transcripts),
                "post_processing_time": 0.0
            }
            
            logger.debug(f"Response post-processing completed: confidence={confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Response post-processing failed: {e}")
            return {
                "processed_response": response,
                "confidence": 0.0,
                "source_count": 0,
                "post_processing_time": 0.0,
                "error": str(e)
            }
    
    def _extract_source_info(self, transcripts: List[Dict[str, Any]]) -> str:
        """
        Extract source information from transcripts.
        
        Args:
            transcripts: List of transcript dictionaries.
            
        Returns:
            Formatted source information string.
        """
        try:
            sources = []
            for transcript in transcripts[:3]:  # Limit to first 3 sources
                speaker = transcript.get("speaker", "Unknown")
                timestamp = transcript.get("timestamp", "Unknown")
                similarity = transcript.get("similarity", 0.0)
                sources.append(f"{speaker} ({timestamp}, similarity: {similarity:.2f})")
            
            return "; ".join(sources)
            
        except Exception as e:
            logger.error(f"Source info extraction failed: {e}")
            return "Source information unavailable"
    
    def _calculate_confidence(self, transcripts: List[Dict[str, Any]], query: str) -> float:
        """
        Calculate confidence score for the response.
        
        Args:
            transcripts: Retrieved transcripts.
            query: Original query.
            
        Returns:
            Confidence score between 0 and 1.
        """
        try:
            if not transcripts:
                return 0.0
            
            # Calculate average similarity
            similarities = [t.get("similarity", 0.0) for t in transcripts if "similarity" in t]
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
            else:
                avg_similarity = 0.5  # Default confidence
            
            # Adjust based on number of sources
            source_factor = min(len(transcripts) / 5.0, 1.0)  # Cap at 5 sources
            
            # Combine factors
            confidence = (avg_similarity * 0.7) + (source_factor * 0.3)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def process_query(
        self,
        query: str,
        limit: int = 10,
        excluded_speakers: Optional[List[str]] = None,
        min_similarity: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_debug: bool = False
    ) -> Dict[str, Any]:
        """
        Process a complete query through the RAG pipeline.
        
        Args:
            query: User query.
            limit: Maximum number of results to retrieve.
            excluded_speakers: List of speaker names to exclude.
            min_similarity: Minimum similarity score threshold.
            temperature: Generation temperature.
            max_tokens: Maximum response tokens.
            include_debug: Whether to include debug information.
            
        Returns:
            Dictionary containing complete pipeline results.
        """
        try:
            start_time = time.time()
            logger.info(f"Processing query through RAG pipeline: '{query[:50]}...'")
            
            # Step 1: Preprocess query
            preprocessing_result = self.preprocess_query(query)
            processed_query = preprocessing_result["processed_query"]
            
            # Step 2: Retrieve context
            retrieval_result = self.retrieve_context(
                query=processed_query,
                limit=limit,
                excluded_speakers=excluded_speakers,
                min_similarity=min_similarity
            )
            
            # Step 3: Generate response
            generation_result = self.generate_response(
                query=processed_query,
                context=retrieval_result["context"],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Step 4: Post-process response
            post_processing_result = self.post_process_response(
                response=generation_result["response"],
                query=processed_query,
                context=retrieval_result["context"],
                transcripts=retrieval_result["transcripts"]
            )
            
            # Compile final result
            result = {
                "query": query,
                "processed_query": processed_query,
                "response": post_processing_result["processed_response"],
                "confidence": post_processing_result["confidence"],
                "total_time": time.time() - start_time,
                "error": None
            }
            
            # Add debug information if requested
            if include_debug or self.debug_mode:
                result["debug"] = {
                    "preprocessing": preprocessing_result,
                    "retrieval": retrieval_result,
                    "generation": generation_result,
                    "post_processing": post_processing_result
                }
            
            logger.info(f"RAG pipeline completed: {result['total_time']:.2f}s, confidence: {result['confidence']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return {
                "query": query,
                "processed_query": query,
                "response": "I'm sorry, I couldn't process your query at this time.",
                "confidence": 0.0,
                "total_time": 0.0,
                "error": str(e)
            }
    
    def preview_retrieval(
        self,
        query: str,
        limit: int = 5,
        excluded_speakers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Preview retrieval results without generating a response.
        
        Args:
            query: User query.
            limit: Maximum number of results to retrieve.
            excluded_speakers: List of speaker names to exclude.
            
        Returns:
            Dictionary containing retrieval preview.
        """
        try:
            logger.info(f"Previewing retrieval for query: '{query[:50]}...'")
            
            # Preprocess query
            preprocessing_result = self.preprocess_query(query)
            
            # Retrieve context
            retrieval_result = self.retrieve_context(
                query=preprocessing_result["processed_query"],
                limit=limit,
                excluded_speakers=excluded_speakers
            )
            
            # Format preview
            preview = {
                "query": query,
                "processed_query": preprocessing_result["processed_query"],
                "transcripts": retrieval_result["transcripts"],
                "context_preview": retrieval_result["context"][:500] + "..." if len(retrieval_result["context"]) > 500 else retrieval_result["context"],
                "stats": retrieval_result["stats"],
                "total_time": preprocessing_result["preprocessing_time"] + retrieval_result["retrieval_time"]
            }
            
            logger.info(f"Retrieval preview completed: {len(retrieval_result['transcripts'])} transcripts")
            return preview
            
        except Exception as e:
            logger.error(f"Retrieval preview failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "transcripts": [],
                "context_preview": "",
                "stats": {},
                "total_time": 0.0
            }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary containing pipeline statistics.
        """
        try:
            retriever_stats = self.retriever.get_search_stats([])
            generator_stats = self.generator.get_generation_stats()
            
            stats = {
                "debug_mode": self.debug_mode,
                "retriever": retriever_stats,
                "generator": generator_stats,
                "pipeline_components": {
                    "retriever": "TranscriptRetriever",
                    "generator": "ResponseGenerator"
                }
            }
            
            logger.debug(f"Pipeline stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {e}")
            return {"error": str(e)}
