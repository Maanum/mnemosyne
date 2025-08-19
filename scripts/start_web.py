#!/usr/bin/env python3
"""
Start the Mnemosyne web interface.

This script starts the Flask web application that provides a modern,
Claude-style chat interface for querying the research knowledge base.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main function to start the web interface."""
    print("🌐 Starting Mnemosyne Web Interface...")
    print("=" * 50)
    print("🔗 Open your browser to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        from app import app, initialize_app
        
        # Initialize the application
        if not initialize_app():
            print("❌ Failed to initialize Mnemosyne web interface")
            print("💡 Make sure your database is populated and configuration is correct")
            sys.exit(1)
        
        # Start the web server
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have installed all dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
