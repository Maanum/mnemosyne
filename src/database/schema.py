"""
Weaviate schema management module.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

from .client import get_client

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import weaviate_config

logger = logging.getLogger(__name__)


class SchemaManager:
    """
    Manages Weaviate schema operations.
    
    This class handles schema creation, validation, updates, and resets
    in an idempotent manner.
    """
    
    def __init__(self):
        """Initialize the SchemaManager."""
        self.client = get_client()
        logger.debug("SchemaManager initialized")
    
    def get_schema_config(self) -> Dict[str, Any]:
        """
        Get the default schema configuration for Transcript class.
        
        Returns:
            Schema configuration dictionary.
        """
        return weaviate_config.get_schema_config()
    
    def schema_exists(self, class_name: str) -> bool:
        """
        Check if a schema class exists.
        
        Args:
            class_name: Name of the schema class to check.
            
        Returns:
            True if class exists, False otherwise.
        """
        try:
            with self.client.get_connection() as client:
                schema = client.collections.list_all()
                existing_classes = [collection_name for collection_name in schema]
                return class_name in existing_classes
        except Exception as e:
            logger.error(f"Failed to check if schema {class_name} exists: {e}")
            return False
    
    def create_schema(self, schema_config: Optional[Dict[str, Any]] = None, force: bool = False) -> bool:
        """
        Create the schema class.
        
        Args:
            schema_config: Schema configuration. If None, uses default.
            force: Force recreation even if schema exists.
            
        Returns:
            True if schema created successfully, False otherwise.
        """
        try:
            if schema_config is None:
                schema_config = self.get_schema_config()
            
            class_name = schema_config.get("name") or schema_config.get("class")
            if not class_name:
                raise ValueError("Schema config must contain 'name' field")
            
            logger.info(f"Creating schema for class: {class_name}")
            
            with self.client.get_connection() as client:
                # Check if schema already exists
                if self.schema_exists(class_name):
                    if force:
                        logger.info(f"Schema {class_name} exists, deleting and recreating")
                        self.delete_schema(class_name)
                    else:
                        logger.info(f"Schema {class_name} already exists, skipping creation")
                        return True
                
                # Create the schema
                client.collections.create(**schema_config)
                logger.info(f"Schema {class_name} created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False
    
    def delete_schema(self, class_name: str) -> bool:
        """
        Delete a schema class.
        
        Args:
            class_name: Name of the schema class to delete.
            
        Returns:
            True if schema deleted successfully, False otherwise.
        """
        try:
            logger.info(f"Deleting schema class: {class_name}")
            
            with self.client.get_connection() as client:
                client.collections.delete(class_name)
                logger.info(f"Schema {class_name} deleted successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete schema {class_name}: {e}")
            return False
    
    def delete_all_schemas(self) -> bool:
        """
        Delete all schema classes.
        
        Returns:
            True if all schemas deleted successfully, False otherwise.
        """
        try:
            logger.info("Deleting all schema classes")
            
            with self.client.get_connection() as client:
                # Delete all collections
                collections = client.collections.list_all()
                for collection in collections:
                    client.collections.delete(collection.name)
                logger.info("All schema classes deleted successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete all schemas: {e}")
            return False
    
    def get_schema(self, class_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get schema information.
        
        Args:
            class_name: Name of the schema class. If None, returns all schemas.
            
        Returns:
            Schema information dictionary.
        """
        try:
            with self.client.get_connection() as client:
                if class_name:
                    schema = client.schema.get(class_name)
                else:
                    schema = client.schema.get()
                
                logger.debug(f"Retrieved schema for {class_name or 'all classes'}")
                return schema
                
        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return {"error": str(e)}
    
    def validate_schema(self, class_name: str) -> Dict[str, Any]:
        """
        Validate a schema class.
        
        Args:
            class_name: Name of the schema class to validate.
            
        Returns:
            Validation results dictionary.
        """
        try:
            logger.info(f"Validating schema: {class_name}")
            
            # Check if schema exists
            if not self.schema_exists(class_name):
                return {
                    "valid": False,
                    "error": f"Schema class '{class_name}' does not exist"
                }
            
            # Get schema details
            schema = self.get_schema(class_name)
            if "error" in schema:
                return {
                    "valid": False,
                    "error": schema["error"]
                }
            
            # Basic validation checks
            validation_result = {
                "valid": True,
                "class_name": class_name,
                "properties": [],
                "vectorizer": None,
                "module_config": None,
                "warnings": []
            }
            
            # Extract class information
            if "class" in schema:
                class_info = schema["class"]
                validation_result["vectorizer"] = class_info.get("vectorizer")
                validation_result["module_config"] = class_info.get("moduleConfig")
                
                # Validate properties
                properties = class_info.get("properties", [])
                validation_result["properties"] = [prop.get("name") for prop in properties]
                
                # Check for required properties
                required_props = ["text", "speaker", "timestamp"]
                missing_props = [prop for prop in required_props if prop not in validation_result["properties"]]
                if missing_props:
                    validation_result["warnings"].append(f"Missing properties: {missing_props}")
                
                # Check vectorizer configuration
                if not validation_result["vectorizer"]:
                    validation_result["warnings"].append("No vectorizer configured")
                
            else:
                validation_result["valid"] = False
                validation_result["error"] = "Invalid schema format"
            
            logger.debug(f"Schema validation result: {validation_result}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return {
                "valid": False,
                "error": str(e)
            }
    
    def update_schema(self, class_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing schema class.
        
        Args:
            class_name: Name of the schema class to update.
            updates: Dictionary containing schema updates.
            
        Returns:
            True if schema updated successfully, False otherwise.
        """
        try:
            logger.info(f"Updating schema: {class_name}")
            
            if not self.schema_exists(class_name):
                logger.error(f"Cannot update schema {class_name}: class does not exist")
                return False
            
            with self.client.get_connection() as client:
                # Note: Weaviate doesn't support direct schema updates
                # This would require deleting and recreating the class
                logger.warning(f"Schema updates not supported by Weaviate. Use delete_schema() and create_schema() instead.")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update schema {class_name}: {e}")
            return False
    
    def reset_schema(self, class_name: str) -> bool:
        """
        Reset a schema class (delete and recreate).
        
        Args:
            class_name: Name of the schema class to reset.
            
        Returns:
            True if schema reset successfully, False otherwise.
        """
        try:
            logger.info(f"Resetting schema: {class_name}")
            
            # Delete existing schema
            if self.schema_exists(class_name):
                if not self.delete_schema(class_name):
                    return False
            
            # Create new schema
            return self.create_schema()
            
        except Exception as e:
            logger.error(f"Failed to reset schema {class_name}: {e}")
            return False
    
    def get_schema_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all schemas.
        
        Returns:
            Dictionary containing schema statistics.
        """
        try:
            with self.client.get_connection() as client:
                schema = client.schema.get()
                classes = schema.get("classes", [])
                
                stats = {
                    "total_classes": len(classes),
                    "classes": [],
                    "total_properties": 0
                }
                
                for cls in classes:
                    class_name = cls.get("class", "unknown")
                    properties = cls.get("properties", [])
                    
                    class_stats = {
                        "name": class_name,
                        "properties": len(properties),
                        "vectorizer": cls.get("vectorizer"),
                        "has_module_config": bool(cls.get("moduleConfig"))
                    }
                    
                    stats["classes"].append(class_stats)
                    stats["total_properties"] += len(properties)
                
                logger.debug(f"Schema stats: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get schema stats: {e}")
            return {"error": str(e)}
