"""
Weaviate client management module.
"""

import logging
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

import weaviate
from weaviate.embedded import EmbeddedOptions

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import weaviate_config, openai_config, path_config

logger = logging.getLogger(__name__)

# Global client instance
_client_instance: Optional['WeaviateClient'] = None


class WeaviateClient:
    """
    Manages Weaviate client connection and operations.
    
    This class handles the setup, configuration, and health monitoring
    of the Weaviate vector database client.
    """
    
    def __init__(self, auth_token: Optional[str] = None):
        """
        Initialize the Weaviate client.
        
        Args:
            auth_token: OpenAI API key. If None, uses config default.
        """
        self.auth_token = auth_token or openai_config.api_key
        self.client = None
        self._connection_healthy = False
        self._last_health_check = 0
        self._health_check_interval = 300  # 5 minutes
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Weaviate client with proper configuration."""
        try:
            logger.info("Initializing Weaviate client")
            
            # Get client configuration
            client_config = weaviate_config.get_client_config()
            
            # Create client
            if weaviate_config.embedded:
                logger.info("Using embedded Weaviate instance")
                print("🔧 Starting Weaviate database... (this may take a few seconds)")
                
                # Suppress Weaviate embedded server logs
                weaviate_logger = logging.getLogger('weaviate')
                weaviate_logger.setLevel(logging.ERROR)
                
                # Capture stderr during client creation to suppress Go logs
                from contextlib import redirect_stderr
                from io import StringIO
                
                stderr_capture = StringIO()
                
                try:
                    # Environment variables are set globally in config/settings.py
                    self.client = weaviate.connect_to_embedded()
                    
                    logger.info("Successfully connected to Weaviate")
                    print("✅ Weaviate database ready")
                    
                except Exception as e:
                    logger.error(f"Failed to create Weaviate client: {e}")
                    raise
            else:
                logger.info("Using external Weaviate instance")
                # For external instances, you would need to add URL configuration
                # self.client = weaviate.connect_to_weaviate(url="http://localhost:8080")
                raise NotImplementedError("External Weaviate instances not yet supported")
            
            # Test connection
            self._test_connection()
            logger.info("Weaviate client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            raise RuntimeError(f"Could not initialize Weaviate client: {e}")
    
    def _test_connection(self) -> bool:
        """
        Test the connection to Weaviate.
        
        Returns:
            True if connection is healthy, False otherwise.
        """
        try:
            # Simple health check - try to get collections
            self.client.collections.list_all()
            self._connection_healthy = True
            self._last_health_check = time.time()
            logger.debug("Weaviate connection test successful")
            return True
        except Exception as e:
            self._connection_healthy = False
            logger.error(f"Weaviate connection test failed: {e}")
            return False
    
    def is_healthy(self, force_check: bool = False) -> bool:
        """
        Check if the Weaviate connection is healthy.
        
        Args:
            force_check: Force a new health check regardless of cache.
            
        Returns:
            True if connection is healthy, False otherwise.
        """
        current_time = time.time()
        
        # Use cached result if recent enough
        if not force_check and (current_time - self._last_health_check) < self._health_check_interval:
            return self._connection_healthy
        
        # Perform new health check
        return self._test_connection()
    
    def get_client(self):
        """
        Get the underlying Weaviate client instance.
        
        Returns:
            Weaviate client instance.
        """
        if not self.is_healthy():
            logger.warning("Weaviate connection is unhealthy, attempting to reconnect")
            self._initialize_client()
        
        return self.client
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for safe database operations.
        
        Yields:
            Weaviate client instance.
        """
        try:
            client = self.get_client()
            yield client
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise
    
    def reset_connection(self):
        """Reset the database connection."""
        try:
            logger.info("Resetting Weaviate connection")
            self.client = None
            self._connection_healthy = False
            self._last_health_check = 0
            self._initialize_client()
            logger.info("Weaviate connection reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset Weaviate connection: {e}")
            raise RuntimeError(f"Connection reset failed: {e}")
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the Weaviate server.
        
        Returns:
            Dictionary containing server information.
        """
        try:
            with self.get_connection() as client:
                # Get server meta information
                meta = client.get_meta()
                
                info = {
                    "version": meta.get("version", "unknown"),
                    "modules": meta.get("modules", []),
                    "hostname": meta.get("hostname", "unknown"),
                    "modules_installed": len(meta.get("modules", [])),
                    "connection_healthy": self._connection_healthy,
                    "last_health_check": self._last_health_check
                }
                
                logger.debug(f"Server info: {info}")
                return info
                
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            return {"error": str(e)}
    
    def ping(self) -> bool:
        """
        Simple ping to test connectivity.
        
        Returns:
            True if ping successful, False otherwise.
        """
        try:
            with self.get_connection() as client:
                # Simple ping operation
                client.get_meta()
                return True
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            return False


def get_client(auth_token: Optional[str] = None) -> WeaviateClient:
    """
    Get or create a global Weaviate client instance.
    
    Args:
        auth_token: OpenAI API key. If None, uses config default.
        
    Returns:
        WeaviateClient instance.
    """
    global _client_instance
    
    if _client_instance is None:
        logger.info("Creating new global Weaviate client instance")
        _client_instance = WeaviateClient(auth_token=auth_token)
    else:
        logger.debug("Using existing global Weaviate client instance")
    
    return _client_instance


def reset_global_client():
    """Reset the global client instance."""
    global _client_instance
    
    if _client_instance is not None:
        logger.info("Resetting global Weaviate client instance")
        _client_instance = None
    else:
        logger.debug("No global client instance to reset")


def is_connected() -> bool:
    """
    Check if the global client is connected and healthy.
    
    Returns:
        True if connected and healthy, False otherwise.
    """
    try:
        client = get_client()
        return client.is_healthy()
    except Exception as e:
        logger.error(f"Connection check failed: {e}")
        return False
