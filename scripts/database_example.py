#!/usr/bin/env python3
"""
Example script demonstrating the new modular database system.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import setup_logging
from database import WeaviateClient, get_client, SchemaManager, DataIngester
from config.settings import path_config


def main():
    """Example usage of the database modules."""
    
    # Set up logging
    setup_logging(level="INFO")
    
    print("=== Database System Example ===\n")
    
    # Example 1: Client Management
    print("1. Client Management")
    print("-" * 20)
    
    # Get global client instance
    client = get_client()
    print(f"✅ Global client obtained")
    
    # Check connection health
    if client.is_healthy():
        print(f"✅ Connection is healthy")
        
        # Get server info
        server_info = client.get_server_info()
        print(f"   Server version: {server_info.get('version', 'unknown')}")
        print(f"   Modules installed: {server_info.get('modules_installed', 0)}")
    else:
        print(f"❌ Connection is unhealthy")
    
    # Example 2: Schema Management
    print("\n2. Schema Management")
    print("-" * 20)
    
    schema_manager = SchemaManager()
    
    # Check if schema exists
    if schema_manager.schema_exists("Transcript"):
        print(f"✅ Schema 'Transcript' exists")
        
        # Validate schema
        validation = schema_manager.validate_schema("Transcript")
        if validation["valid"]:
            print(f"✅ Schema validation passed")
            print(f"   Properties: {validation.get('properties', [])}")
        else:
            print(f"❌ Schema validation failed: {validation.get('error')}")
    else:
        print(f"❌ Schema 'Transcript' does not exist")
        
        # Create schema
        print("Creating schema...")
        if schema_manager.create_schema():
            print(f"✅ Schema created successfully")
        else:
            print(f"❌ Schema creation failed")
    
    # Get schema statistics
    schema_stats = schema_manager.get_schema_stats()
    print(f"   Total classes: {schema_stats.get('total_classes', 0)}")
    
    # Example 3: Data Ingestion
    print("\n3. Data Ingestion")
    print("-" * 20)
    
    ingester = DataIngester()
    
    # Check if source data directory exists
    source_dir = path_config.source_data_dir
    if source_dir.exists():
        csv_files = list(source_dir.glob("*.csv"))
        if csv_files:
            print(f"Found {len(csv_files)} CSV files in {source_dir}")
            
            # Example: Ingest data (commented out for safety)
            print("To ingest data, uncomment the following code:")
            print("""
            # result = ingester.ingest_csv_directory(
            #     directory=source_dir,
            #     batch_size=50,
            #     show_progress=True,
            #     validate_data=True
            # )
            # 
            # if 'error' not in result:
            #     print(f"✅ Ingestion completed:")
            #     print(f"   Total rows: {result['total_rows']}")
            #     print(f"   Successful: {result['successful_rows']}")
            #     print(f"   Failed: {result['failed_rows']}")
            # else:
            #     print(f"❌ Ingestion failed: {result['error']}")
            """)
        else:
            print(f"No CSV files found in {source_dir}")
    else:
        print(f"Source data directory does not exist: {source_dir}")
    
    # Example 4: Individual Component Usage
    print("\n4. Individual Component Usage")
    print("-" * 20)
    print("You can also use the components individually:")
    print("""
    # Client operations
    client = get_client()
    is_healthy = client.is_healthy()
    server_info = client.get_server_info()
    
    # Schema operations
    schema_manager = SchemaManager()
    schema_exists = schema_manager.schema_exists("Transcript")
    validation = schema_manager.validate_schema("Transcript")
    
    # Data ingestion
    ingester = DataIngester()
    df = ingester.combine_csv_files("./source_data")
    result = ingester.ingest_dataframe(df, batch_size=100)
    """)
    
    # Example 5: Error Handling
    print("\n5. Error Handling")
    print("-" * 20)
    print("All modules include comprehensive error handling:")
    print("""
    try:
        client = get_client()
        if not client.is_healthy():
            print("Connection unhealthy, attempting reset...")
            client.reset_connection()
    except Exception as e:
        print(f"Client error: {e}")
    
    try:
        schema_manager = SchemaManager()
        if not schema_manager.create_schema():
            print("Schema creation failed")
    except Exception as e:
        print(f"Schema error: {e}")
    
    try:
        ingester = DataIngester()
        result = ingester.ingest_csv_directory("./source_data")
        if 'error' in result:
            print(f"Ingestion error: {result['error']}")
    except Exception as e:
        print(f"Ingestion error: {e}")
    """)


if __name__ == "__main__":
    main()
