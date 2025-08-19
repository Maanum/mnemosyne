"""
Data ingestion module for loading data into Weaviate.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from tqdm import tqdm

from .client import get_client
from .schema import SchemaManager
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import path_config

logger = logging.getLogger(__name__)


class DataIngester:
    """
    Handles data ingestion into Weaviate vector database.
    
    This class manages the process of loading and validating data from various
    sources (primarily CSV) into Weaviate with support for batch processing
    and progress tracking.
    """
    
    def __init__(self, class_name: str = "Transcript"):
        """
        Initialize the DataIngester.
        
        Args:
            class_name: Name of the Weaviate class to ingest data into.
        """
        self.class_name = class_name
        self.client = get_client()
        self.schema_manager = SchemaManager()
        logger.info(f"DataIngester initialized for class: {class_name}")
    
    def combine_csv_files(self, directory: Union[str, Path]) -> pd.DataFrame:
        """
        Combine multiple CSV files into a single DataFrame.
        
        Args:
            directory: Directory containing CSV files.
            
        Returns:
            Combined DataFrame.
        """
        try:
            directory = Path(directory)
            logger.info(f"Combining CSV files from: {directory}")
            
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
            
            # Find all CSV files
            csv_files = list(directory.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {directory}")
            
            logger.info(f"Found {len(csv_files)} CSV files")
            
            # Read and combine all CSV files
            all_data = []
            for csv_file in csv_files:
                logger.debug(f"Reading CSV file: {csv_file}")
                try:
                    df = pd.read_csv(csv_file)
                    all_data.append(df)
                    logger.debug(f"Loaded {len(df)} rows from {csv_file.name}")
                except Exception as e:
                    logger.error(f"Failed to read {csv_file}: {e}")
                    continue
            
            if not all_data:
                raise ValueError("No valid CSV files could be read")
            
            # Combine all DataFrames
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Clean data
            combined_df = self._clean_dataframe(combined_df)
            
            logger.info(f"Combined {len(combined_df)} total rows from {len(all_data)} files")
            return combined_df
            
        except Exception as e:
            logger.error(f"Failed to combine CSV files: {e}")
            raise RuntimeError(f"CSV combination failed: {e}")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare DataFrame for ingestion.
        
        Args:
            df: Raw DataFrame to clean.
            
        Returns:
            Cleaned DataFrame.
        """
        try:
            logger.debug("Cleaning DataFrame")
            
            # Replace infinities with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with appropriate defaults
            df = df.fillna({
                'Text': '',
                'Speaker': 'Unknown',
                'Timestamp': '00:00:00'
            })
            
            # Ensure text column is string type
            if 'Text' in df.columns:
                df['Text'] = df['Text'].astype(str)
            
            # Ensure speaker column is string type
            if 'Speaker' in df.columns:
                df['Speaker'] = df['Speaker'].astype(str)
            
            # Ensure timestamp column is string type
            if 'Timestamp' in df.columns:
                df['Timestamp'] = df['Timestamp'].astype(str)
            
            logger.debug(f"DataFrame cleaned: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to clean DataFrame: {e}")
            raise RuntimeError(f"DataFrame cleaning failed: {e}")
    
    def validate_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Validate a single data row.
        
        Args:
            row: Pandas Series representing a data row.
            
        Returns:
            Validation result dictionary.
        """
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check required fields
            required_fields = ['Text', 'Speaker', 'Timestamp']
            for field in required_fields:
                if field not in row.index:
                    validation_result["errors"].append(f"Missing required field: {field}")
                    validation_result["valid"] = False
                elif pd.isna(row[field]) or str(row[field]).strip() == '':
                    validation_result["errors"].append(f"Empty required field: {field}")
                    validation_result["valid"] = False
            
            # Validate text content
            if 'Text' in row.index and not pd.isna(row['Text']):
                text = str(row['Text']).strip()
                if len(text) == 0:
                    validation_result["warnings"].append("Empty text content")
                elif len(text) > 10000:  # Arbitrary limit
                    validation_result["warnings"].append("Text content very long")
            
            # Validate timestamp format (basic check)
            if 'Timestamp' in row.index and not pd.isna(row['Timestamp']):
                timestamp = str(row['Timestamp'])
                if not any(char.isdigit() for char in timestamp):
                    validation_result["warnings"].append("Timestamp may not be in expected format")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Row validation failed: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {e}"],
                "warnings": []
            }
    
    def add_to_weaviate(self, text: str, speaker: str, timestamp: str) -> bool:
        """
        Add a single record to Weaviate.
        
        Args:
            text: Text content.
            speaker: Speaker name.
            timestamp: Timestamp.
            
        Returns:
            True if successfully added, False otherwise.
        """
        try:
            with self.client.get_connection() as client:
                client.collections.get(self.class_name).data.insert({
                    "text": text,
                    "speaker": speaker,
                    "timestamp": timestamp
                })
                return True
                
        except Exception as e:
            logger.error(f"Failed to add data to Weaviate: {e}")
            return False
    
    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 100,
        show_progress: bool = True,
        validate_data: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest a DataFrame into Weaviate.
        
        Args:
            df: DataFrame to ingest.
            batch_size: Number of records to process in each batch.
            show_progress: Whether to show progress bar.
            validate_data: Whether to validate data before ingestion.
            
        Returns:
            Dictionary containing ingestion results and statistics.
        """
        try:
            logger.info(f"Starting data ingestion: {len(df)} rows, batch_size={batch_size}")
            
            # Ensure schema exists
            if not self.schema_manager.schema_exists(self.class_name):
                logger.info(f"Schema {self.class_name} does not exist, creating it")
                if not self.schema_manager.create_schema():
                    raise RuntimeError(f"Failed to create schema {self.class_name}")
            
            # Initialize results
            results = {
                "total_rows": len(df),
                "processed_rows": 0,
                "successful_rows": 0,
                "failed_rows": 0,
                "validation_errors": 0,
                "insertion_errors": 0,
                "errors": [],
                "warnings": []
            }
            
            # Process data in batches
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            if show_progress:
                pbar = tqdm(total=len(df), desc="Ingesting data", unit="row")
            
            for batch_start in range(0, len(df), batch_size):
                batch_end = min(batch_start + batch_size, len(df))
                batch_df = df.iloc[batch_start:batch_end]
                
                logger.debug(f"Processing batch {batch_start//batch_size + 1}/{total_batches}: rows {batch_start}-{batch_end-1}")
                
                for index, row in batch_df.iterrows():
                    try:
                        # Validate row if requested
                        if validate_data:
                            validation = self.validate_row(row)
                            if not validation["valid"]:
                                results["validation_errors"] += 1
                                results["errors"].extend(validation["errors"])
                                logger.warning(f"Row {index} validation failed: {validation['errors']}")
                                continue
                            
                            results["warnings"].extend(validation["warnings"])
                        
                        # Extract data
                        text = str(row.get('Text', '')).strip()
                        speaker = str(row.get('Speaker', 'Unknown')).strip()
                        timestamp = str(row.get('Timestamp', '00:00:00')).strip()
                        
                        # Add to Weaviate
                        if self.add_to_weaviate(text, speaker, timestamp):
                            results["successful_rows"] += 1
                        else:
                            results["insertion_errors"] += 1
                            results["failed_rows"] += 1
                        
                        results["processed_rows"] += 1
                        
                        if show_progress:
                            pbar.update(1)
                            
                    except Exception as e:
                        results["failed_rows"] += 1
                        results["errors"].append(f"Row {index}: {e}")
                        logger.error(f"Failed to process row {index}: {e}")
                        
                        if show_progress:
                            pbar.update(1)
            
            if show_progress:
                pbar.close()
            
            # Get final object count
            try:
                with self.client.get_connection() as client:
                    collection = client.collections.get(self.class_name)
                    object_count = collection.aggregate.over_all(total_count=True)
                    final_count = object_count.total_count if hasattr(object_count, 'total_count') else 0
                    results["final_object_count"] = final_count
            except Exception as e:
                logger.warning(f"Could not get final object count: {e}")
                results["final_object_count"] = "unknown"
            
            logger.info(f"Ingestion completed: {results['successful_rows']} successful, {results['failed_rows']} failed")
            return results
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            return {
                "error": str(e),
                "total_rows": len(df) if 'df' in locals() else 0,
                "processed_rows": 0,
                "successful_rows": 0,
                "failed_rows": len(df) if 'df' in locals() else 0
            }
    
    def ingest_csv_directory(
        self,
        directory: Union[str, Path],
        batch_size: int = 100,
        show_progress: bool = True,
        validate_data: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest all CSV files from a directory.
        
        Args:
            directory: Directory containing CSV files.
            batch_size: Number of records to process in each batch.
            show_progress: Whether to show progress bar.
            validate_data: Whether to validate data before ingestion.
            
        Returns:
            Dictionary containing ingestion results and statistics.
        """
        try:
            logger.info(f"Starting CSV directory ingestion: {directory}")
            
            # Combine CSV files
            df = self.combine_csv_files(directory)
            
            # Ingest the combined data
            return self.ingest_dataframe(df, batch_size, show_progress, validate_data)
            
        except Exception as e:
            logger.error(f"CSV directory ingestion failed: {e}")
            return {
                "error": str(e),
                "total_rows": 0,
                "processed_rows": 0,
                "successful_rows": 0,
                "failed_rows": 0
            }
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get statistics about ingested data.
        
        Returns:
            Dictionary containing ingestion statistics.
        """
        try:
            with self.client.get_connection() as client:
                collection = client.collections.get(self.class_name)
                
                # Get object count
                object_count = collection.aggregate.over_all(total_count=True)
                total_count = object_count.total_count if hasattr(object_count, 'total_count') else 0
                
                # Get sample data
                sample_response = collection.query.fetch_objects(
                    limit=5,
                    return_properties=["text", "speaker", "timestamp"]
                )
                sample_data = []
                for obj in sample_response.objects:
                    sample_data.append({
                        "text": obj.properties.get("text", ""),
                        "speaker": obj.properties.get("speaker", ""),
                        "timestamp": obj.properties.get("timestamp", "")
                    })
                
                stats = {
                    "class_name": self.class_name,
                    "total_objects": total_count,
                    "sample_data": sample_data,
                    "schema_valid": self.schema_manager.validate_schema(self.class_name)["valid"]
                }
                
                logger.debug(f"Ingestion stats: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            return {"error": str(e)}
