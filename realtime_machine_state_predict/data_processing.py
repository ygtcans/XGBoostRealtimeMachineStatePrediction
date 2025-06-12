# realtime_machine_state_predict/data_processing.py

import pandas as pd
import numpy as np
import json
import warnings
import logging
from typing import Union, List, Tuple
from sklearn.preprocessing import OneHotEncoder

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WebSocketDataConverter:
    """
    Handles conversion of raw WebSocket JSON messages to pandas DataFrame.
    Follows Single Responsibility Principle by focusing solely on data transformation.
    """
    
    def convert(self, ws_data: str) -> Union[pd.DataFrame, None]:
        """
        Convert raw WebSocket JSON string to pandas DataFrame.
        
        Args:
            ws_data (str): Raw JSON string received from WebSocket
            
        Returns:
            Union[pd.DataFrame, None]: DataFrame containing sensor data or None if conversion fails
        """
        try:
            data = json.loads(ws_data)
            
            # Ensure data is a list of dictionaries for DataFrame creation
            if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
                # If it's a dictionary of dictionaries, convert values to a list
                data = list(data.values())
            elif not isinstance(data, list):
                # If it's a single dictionary, wrap it in a list
                data = [data]
            
            df = pd.DataFrame(data)
            
            if df.empty:
                logger.warning("Converted DataFrame is empty.")
                return None
            
            # Rename 'state' column to 'actual_state' if it exists in the incoming data
            if 'state' in df.columns:
                df.rename(columns={'state': 'actual_state'}, inplace=True)

            # Validate presence of *at least* basic required columns.
            # More extensive validation will happen during preprocessing based on features_to_keep.
            basic_required_columns = ['machine_id', 'timestamp', 'temperature', 'pressure', 
                                      'energy_consumption', 'vibration', 'product_type']
            
            missing_basic_columns = set(basic_required_columns) - set(df.columns)
            if missing_basic_columns:
                logger.error(f"Missing essential columns in incoming data: {missing_basic_columns}. Skipping data.")
                return None
            
            return df
            
        except json.JSONDecodeError:
            logger.error("Incoming WebSocket message is not valid JSON.")
            return None
        except Exception as e:
            logger.error(f"Data conversion error: {e}", exc_info=True)
            return None

class DataPreprocessor:
    """
    Manages all preprocessing steps required before feeding data to the model.
    Handles feature engineering, encoding, and ensuring feature consistency.
    """
    
    def __init__(self, product_types: List[str], features_expected: List[str]):
        """
        Initialize preprocessor with product types and expected features.
        
        Args:
            product_types (list): List of valid product types for OneHot encoding.
            features_expected (list): List of feature names expected by the model in a specific order.
        """
        self.product_types = product_types
        # Initialize OneHotEncoder for product types with unknown value handling
        self.ohe_encoder = OneHotEncoder(
            categories=[product_types],
            handle_unknown='ignore',
            sparse_output=False
        )
        # Fit the encoder on the product types to ensure all categories are known
        self.ohe_encoder.fit(np.array(product_types).reshape(-1, 1))
        self.features_expected = features_expected
        
        logger.info(f"OneHotEncoder initialized with categories: {product_types}")
        logger.info(f"Model expects features (count: {len(features_expected)}): {features_expected[:5]}...{features_expected[-5:]}") # Log a snippet

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to raw DataFrame to prepare it for model inference.
        
        Args:
            df (pd.DataFrame): Raw DataFrame from WebSocket.
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame with exactly the expected features, or empty DataFrame if critical issues.
        """
        if df.empty:
            logger.warning("Received empty DataFrame for preprocessing.")
            return pd.DataFrame()

        df_processed = df.copy()

        # Convert timestamp to datetime and handle invalid values
        if 'timestamp' in df_processed.columns:
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce', utc=True)
            initial_rows = df_processed.shape[0]
            df_processed.dropna(subset=['timestamp'], inplace=True)
            if df_processed.shape[0] < initial_rows:
                logger.warning(f"Removed {initial_rows - df_processed.shape[0]} rows due to invalid timestamps.")
        else:
            logger.error("Timestamp column not found in data. Cannot perform time-based feature engineering.")
            return pd.DataFrame()
        
        if df_processed.empty:
            logger.warning("DataFrame empty after timestamp processing.")
            return pd.DataFrame()

        # Feature engineering from timestamp
        df_processed['hour_of_day'] = df_processed['timestamp'].dt.hour
        df_processed['is_weekend'] = (df_processed['timestamp'].dt.dayofweek >= 5).astype(int)

        # Process sensor features - convert to numeric and handle missing values
        sensor_features = ['temperature', 'pressure', 'energy_consumption', 'vibration']
        for feature in sensor_features:
            if feature in df_processed.columns:
                df_processed[feature] = pd.to_numeric(df_processed[feature], errors='coerce')
                # Fill NaN with median if available, else 0.0 (e.g., for all NaNs)
                df_processed[feature] = df_processed[feature].fillna(
                    df_processed[feature].median() if df_processed[feature].count() > 0 else 0.0
                )
            else:
                df_processed[feature] = 0.0 # Add column with default 0 if missing
                logger.warning(f"Sensor feature '{feature}' not found in incoming data, added with default value 0.0.")

            # Handle noisy versions of sensor features as they were generated during training
            # Ensure consistency even if noisy data isn't directly streamed
            noisy_feature_name = f'{feature}_noisy'
            if noisy_feature_name not in df_processed.columns:
                 # If noisy feature not present, use original sensor value (training used original + noise)
                df_processed[noisy_feature_name] = df_processed[feature]
                # logger.debug(f"Created '{noisy_feature_name}' from '{feature}' as it was missing.")
            else:
                df_processed[noisy_feature_name] = pd.to_numeric(df_processed[noisy_feature_name], errors='coerce').fillna(0.0)

        # Apply One-Hot Encoding to product_type
        if 'product_type' in df_processed.columns:
            try:
                # Ensure product_type is treated as categorical for OHE
                product_type_series = df_processed['product_type'].astype(str)
                ohe_transformed = self.ohe_encoder.transform(product_type_series.values.reshape(-1, 1))
                ohe_column_names = self.ohe_encoder.get_feature_names_out(['product_type'])

                ohe_df = pd.DataFrame(
                    ohe_transformed,
                    columns=ohe_column_names,
                    index=df_processed.index
                )
                # Concatenate OHE features and drop the original 'product_type'
                df_processed = pd.concat([df_processed.drop(columns=['product_type']), ohe_df], axis=1)
            except Exception as e:
                logger.error(f"One-Hot Encoding failed for product_type: {e}", exc_info=True)
                # Fallback: Create zero-filled OHE columns if OHE fails
                for col_name in self.ohe_encoder.get_feature_names_out(['product_type']):
                    if col_name not in df_processed.columns:
                        df_processed[col_name] = 0.0
        else:
            logger.warning("Product_type column not found for OneHot Encoding. Adding zero-filled OHE columns.")
            # If 'product_type' column is missing, create zero-filled columns for all OHE features
            for col_name in self.ohe_encoder.get_feature_names_out(['product_type']):
                df_processed[col_name] = 0.0
        
        # Final step: Ensure all expected features are present AND in the correct order
        final_preprocessed_data = {}
        for col in self.features_expected:
            if col in df_processed.columns:
                final_preprocessed_data[col] = pd.to_numeric(df_processed[col], errors='coerce')
            else:
                final_preprocessed_data[col] = 0.0 # Default value if feature is missing after all steps
                logger.warning(f"Expected feature '{col}' not found or created. Added with default value 0.0.")
            
            # Fill NaN values in the final numeric columns with median (or 0 if all NaNs)
            if pd.api.types.is_numeric_dtype(pd.Series(final_preprocessed_data[col])): # Check dtype of the series
                final_preprocessed_data[col] = pd.Series(final_preprocessed_data[col]).fillna(
                    pd.Series(final_preprocessed_data[col]).median() if pd.Series(final_preprocessed_data[col]).count() > 0 else 0.0
                )
        
        # Create DataFrame from dictionary, ensuring column order
        final_df = pd.DataFrame(final_preprocessed_data, index=df_processed.index)

        # Reorder columns to exactly match model's expected feature order
        try:
            return final_df[self.features_expected]
        except KeyError as e:
            logger.critical(f"A critical expected feature is missing in the final preprocessed data: {e}. Cannot proceed.")
            return pd.DataFrame()