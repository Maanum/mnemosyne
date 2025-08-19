#!/usr/bin/env python3
"""
Command-line interface for database population.

This script provides a clean CLI for populating the Weaviate database with
transcript data from various sources.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database import DataIngester, SchemaManager, get_client
from utils import setup_logging
from config.settings import path_config


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Populate Weaviate database with transcript data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Populate from default source directory
  python scripts/populate_database.py

  # Populate from specific directory
  python scripts/populate_database.py --input-dir ./my_data

  # Reset schema and populate
  python scripts/populate_database.py --reset-schema --input-dir ./data

  # Populate with custom settings
  python scripts/populate_database.py --batch-size 50 --validate-data --verbose

  # Preview what would be ingested
  python scripts/populate_database.py --dry-run --input-dir ./data

  # Validate existing data
  python scripts/populate_database.py --validate-only
        """
    )
    
    # Input arguments
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/source_data/",
        help="Directory containing CSV files to ingest (default: data/source_data/)"
    )
    
    # Schema options
    parser.add_argument(
        "--reset-schema",
        action="store_true",
        help="Delete and recreate schema before ingestion"
    )
    parser.add_argument(
        "--validate-schema",
        action="store_true",
        help="Validate schema before ingestion"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data without ingesting"
    )
    
    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for ingestion (default: 100)"
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate data before ingestion"
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="Transcript",
        help="Weaviate class name (default: Transcript)"
    )
    
    # Control options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be ingested without actually running"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force ingestion even if validation fails"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: logs/populate_database.log)"
    )
    
    return parser


def validate_database_connection() -> bool:
    """Validate database connection."""
    try:
        client = get_client()
        if client.is_healthy():
            print("✅ Database connection is healthy")
            return True
        else:
            print("❌ Database connection is unhealthy")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return False


def validate_schema(schema_manager: SchemaManager, class_name: str) -> bool:
    """Validate schema."""
    print(f"🔍 Validating schema for class: {class_name}")
    
    validation = schema_manager.validate_schema(class_name)
    
    if validation["valid"]:
        print("✅ Schema validation passed")
        print(f"   Properties: {validation.get('properties', [])}")
        if validation.get("warnings"):
            print(f"   ⚠️  Warnings: {validation['warnings']}")
        return True
    else:
        print("❌ Schema validation failed")
        print(f"   Error: {validation.get('error', 'Unknown error')}")
        return False


def reset_schema(schema_manager: SchemaManager, class_name: str) -> bool:
    """Reset schema."""
    print(f"🔄 Resetting schema for class: {class_name}")
    
    try:
        if schema_manager.delete_schema(class_name):
            print("✅ Schema deleted successfully")
        else:
            print("⚠️  Schema deletion failed or schema didn't exist")
        
        if schema_manager.create_schema():
            print("✅ Schema created successfully")
            return True
        else:
            print("❌ Schema creation failed")
            return False
    except Exception as e:
        print(f"❌ Schema reset failed: {e}")
        return False


def check_directory_for_csv_files(directory: Path) -> tuple[list, list]:
    """Check directory for CSV and non-CSV files."""
    csv_files = []
    other_files = []
    
    for file_path in directory.glob("*"):
        if file_path.is_file():
            if file_path.suffix.lower() == '.csv':
                csv_files.append(file_path)
            else:
                other_files.append(file_path)
    
    return sorted(csv_files), sorted(other_files)


def preview_ingestion(input_dir: str, ingester: DataIngester) -> bool:
    """Preview what would be ingested."""
    input_path = Path(input_dir)
    
    print(f"🔍 Previewing ingestion from: {input_path}")
    
    # Check for CSV files first
    csv_files, other_files = check_directory_for_csv_files(input_path)
    
    if not csv_files:
        if not other_files:
            print(f"❌ No CSV files found in {input_dir}")
            print("   Please add CSV files from your manual review process to the directory")
        else:
            print(f"❌ No CSV files found in {input_dir}")
            print("   Found other files:")
            for file in other_files[:5]:  # Show first 5
                print(f"     - {file.name}")
            if len(other_files) > 5:
                print(f"     ... and {len(other_files) - 5} more")
            print("   Please add CSV files from your manual review process")
        return False
    
    print(f"📋 Found {len(csv_files)} CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"   {i}. {file.name}")
    
    try:
        # Combine CSV files
        df = ingester.combine_csv_files(input_path)
        
        print(f"📊 Data preview:")
        print(f"   Total rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sample data:")
        print(df.head(3).to_string())
        
        # Validate sample data
        print(f"\n🔍 Data validation preview:")
        validation_errors = 0
        validation_warnings = 0
        
        for i, row in df.head(10).iterrows():
            validation = ingester.validate_row(row)
            if not validation["valid"]:
                validation_errors += 1
                print(f"   Row {i}: {validation['errors']}")
            if validation["warnings"]:
                validation_warnings += 1
                print(f"   Row {i} warnings: {validation['warnings']}")
        
        print(f"   Validation errors: {validation_errors}")
        print(f"   Validation warnings: {validation_warnings}")
        
        return validation_errors == 0
        
    except Exception as e:
        print(f"❌ Preview failed: {e}")
        return False


def ingest_data(
    input_dir: str,
    ingester: DataIngester,
    batch_size: int,
    validate_data: bool,
    dry_run: bool = False
) -> bool:
    """Ingest data into database."""
    input_path = Path(input_dir)
    
    print(f"📥 Ingesting data from: {input_path}")
    print(f"   Batch size: {batch_size}")
    print(f"   Validate data: {validate_data}")
    
    if dry_run:
        print("🔍 Dry run mode - no data will be ingested")
        return True
    
    try:
        result = ingester.ingest_csv_directory(
            directory=input_path,
            batch_size=batch_size,
            show_progress=True,
            validate_data=validate_data
        )
        
        if "error" in result:
            print(f"❌ Ingestion failed: {result['error']}")
            return False
        
        # Print results
        print(f"\n📊 Ingestion Results:")
        print(f"   ✅ Successful: {result.get('successful_rows', 0)}")
        print(f"   ❌ Failed: {result.get('failed_rows', 0)}")
        print(f"   📊 Total processed: {result.get('processed_rows', 0)}")
        print(f"   🔍 Validation errors: {result.get('validation_errors', 0)}")
        print(f"   📝 Insertion errors: {result.get('insertion_errors', 0)}")
        print(f"   🎯 Final object count: {result.get('final_object_count', 'unknown')}")
        
        # Show warnings if any
        if result.get("warnings"):
            print(f"   ⚠️  Warnings: {result['warnings']}")
        
        return result.get("failed_rows", 0) == 0
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False


def show_database_stats(ingester: DataIngester) -> None:
    """Show database statistics."""
    print(f"\n📊 Database Statistics:")
    
    try:
        stats = ingester.get_ingestion_stats()
        
        if "error" in stats:
            print(f"   ❌ Could not retrieve stats: {stats['error']}")
            return
        
        print(f"   📁 Class: {stats.get('class_name', 'unknown')}")
        print(f"   📊 Total objects: {stats.get('total_objects', 0)}")
        print(f"   ✅ Schema valid: {stats.get('schema_valid', False)}")
        
        # Show sample data
        sample_data = stats.get("sample_data", [])
        if sample_data:
            print(f"   📝 Sample data:")
            for i, sample in enumerate(sample_data[:3], 1):
                speaker = sample.get("speaker", "Unknown")
                timestamp = sample.get("timestamp", "Unknown")
                text = sample.get("text", "")[:100]
                print(f"      {i}. {speaker} ({timestamp}): {text}...")
        
    except Exception as e:
        print(f"   ❌ Failed to get stats: {e}")


def main():
    """Main function."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = "INFO"
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "ERROR"
    
    log_file = args.log_file or path_config.project_root / "logs" / "populate_database.log"
    setup_logging(level=log_level, log_file=log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting database population")
    
    # Print header
    print(f"\n🗄️  MNEMOSYNE DATABASE POPULATOR")
    print(f"{'='*50}")
    
    # Validate database connection
    if not validate_database_connection():
        sys.exit(1)
    
    # Initialize components
    try:
        schema_manager = SchemaManager()
        ingester = DataIngester(class_name=args.class_name)
        logger.info("Components initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        sys.exit(1)
    
    # Handle schema operations
    if args.reset_schema:
        if not reset_schema(schema_manager, args.class_name):
            sys.exit(1)
    elif args.validate_schema:
        if not validate_schema(schema_manager, args.class_name):
            if not args.force:
                sys.exit(1)
    
    # Determine input directory
    input_dir = args.input_dir
    input_path = Path(input_dir)
    
    print(f"📁 Input directory: {input_dir}")
    
    if not input_path.exists():
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        print("   Please create the directory or specify a different path")
        sys.exit(1)
    
    # Preview ingestion
    if args.dry_run or args.validate_only:
        if not preview_ingestion(input_dir, ingester):
            if not args.force:
                sys.exit(1)
    
    if args.validate_only:
        print("🔍 Validation only mode - no data ingested")
        show_database_stats(ingester)
        sys.exit(0)
    
    # Ingest data
    success = False
    
    try:
        success = ingest_data(
            input_dir=input_dir,
            ingester=ingester,
            batch_size=args.batch_size,
            validate_data=args.validate_data,
            dry_run=args.dry_run
        )
    
    except KeyboardInterrupt:
        print("\n⚠️  Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    
    # Show final statistics
    if not args.dry_run:
        show_database_stats(ingester)
    
    # Exit with appropriate code
    if success:
        print("\n🎉 Database population completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Database population completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
