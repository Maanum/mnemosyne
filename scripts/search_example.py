#!/usr/bin/env python3
"""
Example script demonstrating the new modular search system.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import setup_logging
from search import TranscriptRetriever, ResponseGenerator, RAGPipeline
from database import get_client


def main():
    """Example usage of the search modules."""
    
    # Set up logging
    setup_logging(level="INFO")
    
    print("=== Search System Example ===\n")
    
    # Check database connection
    client = get_client()
    if not client.is_healthy():
        print("❌ Database connection is unhealthy. Please ensure Weaviate is running.")
        return
    
    print("✅ Database connection is healthy\n")
    
    # Example 1: Individual Components
    print("1. Individual Components")
    print("-" * 20)
    
    # Transcript Retriever
    retriever = TranscriptRetriever()
    print("✅ TranscriptRetriever initialized")
    
    # Response Generator
    generator = ResponseGenerator()
    print("✅ ResponseGenerator initialized")
    
    # Example 2: Retrieval Only
    print("\n2. Retrieval Only")
    print("-" * 20)
    
    sample_query = "What did the speakers discuss about technology?"
    print(f"Query: {sample_query}")
    
    # Search for transcripts
    transcripts = retriever.search_transcripts(
        query=sample_query,
        limit=5,
        excluded_speakers=["Kristofer"]  # Example exclusion
    )
    
    if transcripts:
        print(f"✅ Found {len(transcripts)} relevant transcripts")
        
        # Get search statistics
        stats = retriever.get_search_stats(transcripts)
        print(f"   Average similarity: {stats.get('avg_similarity', 0):.3f}")
        print(f"   Speakers found: {', '.join(stats.get('speakers', []))}")
        
        # Format context
        context = retriever.format_context(transcripts, include_similarity=True)
        print(f"   Context length: {len(context)} characters")
        print(f"   Context preview: {context[:200]}...")
    else:
        print("❌ No transcripts found")
    
    # Example 3: Generation Only
    print("\n3. Generation Only")
    print("-" * 20)
    
    if transcripts:
        # Format context for generation
        context = retriever.format_context(transcripts)
        
        # Generate response
        generation_result = generator.generate_response(
            question=sample_query,
            context=context,
            temperature=0.7
        )
        
        if "error" not in generation_result:
            print(f"✅ Response generated successfully")
            print(f"   Model: {generation_result.get('model', 'unknown')}")
            print(f"   Response length: {generation_result.get('response_length', 0)} words")
            print(f"   Response: {generation_result.get('response', '')[:200]}...")
        else:
            print(f"❌ Generation failed: {generation_result.get('error')}")
    else:
        print("Skipping generation (no transcripts available)")
    
    # Example 4: Complete RAG Pipeline
    print("\n4. Complete RAG Pipeline")
    print("-" * 20)
    
    # Create RAG pipeline
    pipeline = RAGPipeline(debug_mode=True)
    print("✅ RAGPipeline initialized with debug mode")
    
    # Process a query through the complete pipeline
    query = "What are the main topics discussed in the interviews?"
    print(f"Query: {query}")
    
    result = pipeline.process_query(
        query=query,
        limit=10,
        excluded_speakers=["Kristofer"],
        include_debug=True
    )
    
    if result.get("error"):
        print(f"❌ Pipeline failed: {result['error']}")
    else:
        print(f"✅ Pipeline completed successfully")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        print(f"   Total time: {result.get('total_time', 0):.2f}s")
        print(f"   Response: {result.get('response', '')[:300]}...")
        
        # Show debug information
        if "debug" in result:
            debug = result["debug"]
            print(f"   Debug info:")
            print(f"     - Retrieval time: {debug.get('retrieval', {}).get('retrieval_time', 0):.2f}s")
            print(f"     - Generation time: {debug.get('generation', {}).get('generation_time', 0):.2f}s")
            print(f"     - Transcripts found: {len(debug.get('retrieval', {}).get('transcripts', []))}")
    
    # Example 5: Retrieval Preview
    print("\n5. Retrieval Preview")
    print("-" * 20)
    
    preview_query = "Tell me about the research findings"
    print(f"Preview query: {preview_query}")
    
    preview = pipeline.preview_retrieval(
        query=preview_query,
        limit=3
    )
    
    if preview.get("error"):
        print(f"❌ Preview failed: {preview['error']}")
    else:
        print(f"✅ Preview completed")
        print(f"   Transcripts found: {len(preview.get('transcripts', []))}")
        print(f"   Context preview: {preview.get('context_preview', '')[:200]}...")
    
    # Example 6: Advanced Usage
    print("\n6. Advanced Usage")
    print("-" * 20)
    print("You can also use the components with custom configurations:")
    print("""
    # Custom retriever with specific settings
    retriever = TranscriptRetriever(class_name="Transcript")
    transcripts = retriever.search_transcripts(
        query="your query",
        limit=20,
        min_similarity=0.7,
        excluded_speakers=["Speaker1", "Speaker2"]
    )
    
    # Custom generator with different model
    generator = ResponseGenerator(model="gpt-3.5-turbo")
    result = generator.generate_response(
        question="your question",
        context="your context",
        temperature=0.5,
        max_tokens=500
    )
    
    # Custom pipeline with dependency injection
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        debug_mode=True
    )
    
    # Process with custom parameters
    result = pipeline.process_query(
        query="your query",
        limit=15,
        min_similarity=0.8,
        temperature=0.3,
        include_debug=True
    )
    """)
    
    # Example 7: Error Handling
    print("\n7. Error Handling")
    print("-" * 20)
    print("All modules include comprehensive error handling:")
    print("""
    try:
        # Test retrieval with invalid query
        transcripts = retriever.search_transcripts("")
        if not transcripts:
            print("No results found for empty query")
    except Exception as e:
        print(f"Retrieval error: {e}")
    
    try:
        # Test generation with empty context
        result = generator.generate_response("test", "")
        if "error" in result:
            print(f"Generation error: {result['error']}")
    except Exception as e:
        print(f"Generation error: {e}")
    
    try:
        # Test pipeline with invalid parameters
        result = pipeline.process_query("", limit=-1)
        if result.get("error"):
            print(f"Pipeline error: {result['error']}")
    except Exception as e:
        print(f"Pipeline error: {e}")
    """)


if __name__ == "__main__":
    main()
