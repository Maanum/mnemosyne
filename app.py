#!/usr/bin/env python3
"""
Mnemosyne Flask Web Application
A modern web interface for querying the research knowledge base.
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flask import Flask, render_template, request, jsonify
import json
import logging
from search.pipeline import RAGPipeline
from database.client import get_client

app = Flask(__name__)
app.secret_key = 'mnemosyne-research-assistant-2025'  # Change this in production

# Initialize components
rag_pipeline = None
db_client = None

def initialize_app():
    """Initialize the RAG pipeline and database client."""
    global rag_pipeline, db_client
    try:
        print("🔧 Initializing Mnemosyne web interface...")
        
        # Initialize database client
        db_client = get_client()
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        
        print("✅ Mnemosyne web interface ready!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False

@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('chat.html')

@app.route('/api/query', methods=['POST'])
def query():
    """Handle user queries and return AI responses."""
    try:
        data = request.get_json()
        user_query = data.get('message', '')
        
        if not user_query.strip():
            return jsonify({'error': 'Empty query'}), 400
        
        if not rag_pipeline:
            return jsonify({'error': 'System not initialized'}), 500
        
        # Get response from RAG pipeline
        response_data = rag_pipeline.process_query(user_query)
        
        # Extract the response text and metadata
        if isinstance(response_data, dict):
            response_text = response_data.get('response', str(response_data))
            metadata = {
                'confidence': response_data.get('confidence', 0.0),
                'time_taken': response_data.get('time_taken', 0.0),
                'sources_count': len(response_data.get('context', []))
            }
        else:
            response_text = str(response_data)
            metadata = {}
        
        return jsonify({
            'response': response_text,
            'query': user_query,
            'metadata': metadata
        })
        
    except Exception as e:
        logging.error(f"Query error: {e}")
        return jsonify({'error': 'Sorry, something went wrong. Please try again.'}), 500

@app.route('/api/health')
def health():
    """Check system health and return status."""
    try:
        if not db_client:
            return jsonify({
                'status': 'unhealthy',
                'error': 'Database client not initialized'
            }), 500
        
        # Check database connection
        is_healthy = db_client.is_healthy()
        
        # Get record count if available
        record_count = 0
        try:
            with db_client.get_connection() as client:
                collection = client.collections.get("Transcript")
                result = collection.aggregate.over_all(total_count=True)
                record_count = result.total_count if hasattr(result, 'total_count') else 0
        except Exception:
            pass  # If we can't get count, that's okay
        
        return jsonify({
            'status': 'healthy' if is_healthy else 'unhealthy',
            'database_connected': is_healthy,
            'transcript_count': record_count,
            'rag_pipeline_ready': rag_pipeline is not None
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize the application
    if not initialize_app():
        print("❌ Failed to start Mnemosyne web interface")
        sys.exit(1)
    
    print("🌐 Starting Mnemosyne Web Interface...")
    print("🔗 Open your browser to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
