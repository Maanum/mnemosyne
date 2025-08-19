"""
Response generation module for LLM-based answer generation.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from openai import OpenAI
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import openai_config, path_config

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Handles LLM response generation for RAG pipeline.
    
    This class manages the process of generating responses using LLMs
    with support for different models, context optimization, and
    prompt management.
    """
    
    def __init__(self, model: str = "gpt-4", prompts_file: Optional[Union[str, Path]] = None):
        """
        Initialize the ResponseGenerator.
        
        Args:
            model: LLM model to use for generation.
            prompts_file: Path to prompts configuration file.
        """
        self.model = model
        self.prompts_file = Path(prompts_file) if prompts_file else path_config.config_dir / "prompts.json"
        self.client = None
        self.prompts_config = None
        
        self._initialize_client()
        self._load_prompts()
        logger.info(f"ResponseGenerator initialized with model: {model}")
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            logger.debug("Initializing OpenAI client")
            self.client = OpenAI(
                api_key=openai_config.api_key,
                organization=openai_config.organization,
                project=openai_config.project
            )
            logger.debug("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise RuntimeError(f"Could not initialize OpenAI client: {e}")
    
    def _load_prompts(self):
        """Load prompts configuration from file."""
        try:
            logger.debug(f"Loading prompts from: {self.prompts_file}")
            
            if not self.prompts_file.exists():
                logger.warning(f"Prompts file not found: {self.prompts_file}")
                self.prompts_config = self._get_default_prompts()
                return
            
            with open(self.prompts_file, "r", encoding="utf-8") as file:
                self.prompts_config = json.load(file)
            
            logger.debug("Prompts configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            self.prompts_config = self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict[str, Any]:
        """Get default prompts configuration."""
        return {
            "system_prompt": {
                "role": "system",
                "content": "You are a research assistant helping the user find relevant passages in a set of transcripts that answer their question. The relevant passages you found will be sent next. When answering the User's questions, you should make references to the specific passages either directly in the text or adding reference in parentheses (such as '(Gilles, 23:11)'. Where important, be sure to indicate cases where all interviewees seem to agree on a topic, and where there are different viewpoints."
            },
            "context_management": {
                "max_context_length": 4000,
                "max_context_tokens": 3000,
                "truncation_strategy": "smart",
                "include_metadata": True
            },
            "response_generation": {
                "max_response_length": 1000,
                "temperature": 0.7,
                "include_sources": True,
                "format_references": True
            }
        }
    
    def get_system_prompt(self) -> Dict[str, str]:
        """
        Get the system prompt for the LLM.
        
        Returns:
            System prompt dictionary.
        """
        return self.prompts_config.get("system_prompt", self._get_default_prompts()["system_prompt"])
    
    def get_generation_config(self) -> Dict[str, Any]:
        """
        Get response generation configuration.
        
        Returns:
            Generation configuration dictionary.
        """
        return self.prompts_config.get("response_generation", {})
    
    def get_context_config(self) -> Dict[str, Any]:
        """
        Get context management configuration.
        
        Returns:
            Context configuration dictionary.
        """
        return self.prompts_config.get("context_management", {})
    
    def optimize_context(
        self,
        context: str,
        max_tokens: Optional[int] = None,
        strategy: str = "smart"
    ) -> str:
        """
        Optimize context for LLM input.
        
        Args:
            context: Raw context string.
            max_tokens: Maximum tokens allowed.
            strategy: Truncation strategy ('smart', 'simple', 'none').
            
        Returns:
            Optimized context string.
        """
        try:
            logger.debug(f"Optimizing context with strategy: {strategy}")
            
            if not context.strip():
                logger.warning("Empty context provided")
                return ""
            
            # Get configuration
            config = self.get_context_config()
            if max_tokens is None:
                max_tokens = config.get("max_context_tokens", 3000)
            
            # Apply truncation strategy
            if strategy == "smart":
                # Smart truncation preserves speaker boundaries
                optimized_context = self._truncate_context_smart(context, max_tokens)
            elif strategy == "simple":
                # Simple word-based truncation
                optimized_context = self._truncate_context_simple(context, max_tokens)
            else:
                # No truncation
                optimized_context = context
            
            logger.debug(f"Context optimization: {len(context.split())} -> {len(optimized_context.split())} tokens")
            return optimized_context
            
        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
            return context
    
    def _truncate_context_smart(self, context: str, max_tokens: int) -> str:
        """
        Smart context truncation that preserves speaker boundaries.
        
        Args:
            context: Context string to truncate.
            max_tokens: Maximum number of tokens.
            
        Returns:
            Truncated context string.
        """
        try:
            tokens = context.split()
            
            if len(tokens) <= max_tokens:
                return context
            
            # Truncate to max_tokens
            truncated_tokens = tokens[:max_tokens]
            truncated_context = " ".join(truncated_tokens)
            
            # Try to find a natural break point (speaker boundary)
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
            
            return truncated_context
            
        except Exception as e:
            logger.error(f"Smart truncation failed: {e}")
            return self._truncate_context_simple(context, max_tokens)
    
    def _truncate_context_simple(self, context: str, max_tokens: int) -> str:
        """
        Simple word-based context truncation.
        
        Args:
            context: Context string to truncate.
            max_tokens: Maximum number of tokens.
            
        Returns:
            Truncated context string.
        """
        try:
            tokens = context.split()
            
            if len(tokens) <= max_tokens:
                return context
            
            truncated_tokens = tokens[:max_tokens]
            truncated_context = " ".join(truncated_tokens)
            
            # Add ellipsis if truncated
            if len(truncated_context) < len(context):
                truncated_context += "..."
            
            return truncated_context
            
        except Exception as e:
            logger.error(f"Simple truncation failed: {e}")
            return context
    
    def generate_response(
        self,
        question: str,
        context: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response using the LLM.
        
        Args:
            question: User question.
            context: Context string from retrieval.
            temperature: Generation temperature.
            max_tokens: Maximum response tokens.
            include_sources: Whether to include source references.
            
        Returns:
            Dictionary containing response and metadata.
        """
        try:
            logger.info(f"Generating response for question: '{question[:50]}...'")
            
            # Get configuration
            config = self.get_generation_config()
            if temperature is None:
                temperature = config.get("temperature", 0.7)
            if max_tokens is None:
                max_tokens = config.get("max_response_length", 1000)
            
            # Optimize context
            optimized_context = self.optimize_context(context)
            
            # Prepare messages
            messages = [
                self.get_system_prompt(),
                {"role": "system", "content": optimized_context},
                {"role": "user", "content": question}
            ]
            
            # Generate response
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_text = completion.choices[0].message.content
            
            # Post-process response
            if include_sources:
                response_text = self._post_process_response(response_text, context)
            
            result = {
                "response": response_text,
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "context_length": len(optimized_context.split()),
                "response_length": len(response_text.split()),
                "usage": completion.usage.model_dump() if completion.usage else None
            }
            
            logger.info(f"Response generated successfully: {len(response_text)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "response": "I'm sorry, I couldn't generate a response at this time.",
                "error": str(e),
                "model": self.model
            }
    
    def _post_process_response(self, response: str, context: str) -> str:
        """
        Post-process the generated response.
        
        Args:
            response: Raw response from LLM.
            context: Original context used.
            
        Returns:
            Post-processed response.
        """
        try:
            # Extract speaker references from context
            import re
            speaker_pattern = r'\(([^,]+),\s*\d{2}:\d{2}:\d{2}\)'
            speakers_in_context = set(re.findall(speaker_pattern, context))
            
            # Check if response mentions speakers from context
            mentioned_speakers = set()
            for speaker in speakers_in_context:
                if speaker.lower() in response.lower():
                    mentioned_speakers.add(speaker)
            
            # Add note if no speakers mentioned
            if speakers_in_context and not mentioned_speakers:
                response += f"\n\nNote: The response is based on transcripts from {', '.join(speakers_in_context)}."
            
            return response
            
        except Exception as e:
            logger.error(f"Response post-processing failed: {e}")
            return response
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the response generator.
        
        Returns:
            Dictionary containing generator statistics.
        """
        try:
            stats = {
                "model": self.model,
                "prompts_loaded": self.prompts_config is not None,
                "system_prompt_length": len(self.get_system_prompt().get("content", "")),
                "generation_config": self.get_generation_config(),
                "context_config": self.get_context_config()
            }
            
            logger.debug(f"Generation stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get generation stats: {e}")
            return {"error": str(e)}
