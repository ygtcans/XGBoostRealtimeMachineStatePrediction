# xgboost_model/data_processing.py
import pandas as pd
import numpy as np
import os
import logging

# Configure logging and warnings as in the original script
# logging.basicConfig(level=..., format='...')
# warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Handles data loading, initial cleaning, and feature creation.
    """
    def __init__(self, config: dict):
        self.config = config

    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        # ... (Mevcut load_and_clean_data metodunun içeriği) ...
        logging.info("--- Data Loading and Initial Cleaning ---")
        if not os.path.exists(file_path):
            logging.error(f"Error: File not found at {file_path}")
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except pd.errors.EmptyDataError:
            logging.error(f"Error: The CSV file at {file_path} is empty.")
            raise pd.errors.EmptyDataError(f"The file '{file_path}' is empty.")
        except Exception as e:
            logging.error(f"Error loading CSV file {file_path}: {e}")
            raise

        df.columns = (df.columns.str.strip()
                      .str.lower()
                      .str.replace(' ', '_')
                      .str.replace(r'[^a-zA-Z0-9_]', '', regex=True))

        for col in self.config["DATETIME_COLUMNS"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                if df[col].isnull().any():
                    logging.warning(f"  Warning: NaNs introduced in '{col}' after datetime conversion.")
            else:
                logging.warning(f"  Warning: Datetime column '{col}' not found in data.")

        initial_rows = df.shape[0]
        df.dropna(subset=self.config["DATETIME_COLUMNS"], inplace=True)
        if df.shape[0] < initial_rows:
            logging.info(f"  Removed {initial_rows - df.shape[0]} rows with missing datetime values.")

        initial_rows = df.shape[0]
        df.drop_duplicates(inplace=True)
        if df.shape[0] < initial_rows:
            logging.info(f"  Removed {initial_rows - df.shape[0]} duplicate rows.")

        df.sort_values(by=['machine_id', 'timestamp'], inplace=True)
        logging.info(f"Initial data loaded and cleaned. Shape: {df.shape}")
        return df


    def remove_outliers_iqr(self, df: pd.DataFrame, columns: list, factor: float = None) -> pd.DataFrame:
        # ... (Mevcut remove_outliers_iqr metodunun içeriği) ...
        factor = factor if factor is not None else self.config["IQR_OUTLIER_FACTOR"]
        df_clean = df.copy()
        outlier_indices = set()

        logging.info(f"  Outlier detection (IQR factor={factor}):")
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR

                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    new_outliers = set(df[outlier_mask].index)
                    outlier_indices.update(new_outliers)
                    logging.info(f"    '{col}': {outlier_count} outliers detected.")
            else:
                logging.warning(f"  Skipping outlier detection for '{col}': Not found or not numeric.")

        if outlier_indices:
            df_clean = df_clean.drop(index=list(outlier_indices))
            logging.info(f"  Total {len(outlier_indices)} outlier rows removed.")
        else:
            logging.info("  No outliers found.")

        return df_clean

    def create_time_and_noisy_features(self, df: pd.DataFrame, noise_level: float = None) -> pd.DataFrame:
        # ... (Mevcut create_time_and_noisy_features metodunun içeriği) ...
        noise_level = noise_level if noise_level is not None else self.config["NOISE_LEVEL_FOR_FEATURES"]
        df_new = df.copy()

        if 'timestamp' in df_new.columns:
            df_new['hour_of_day'] = df_new['timestamp'].dt.hour
            df_new['is_weekend'] = (df_new['timestamp'].dt.dayofweek >= 5).astype(int)
        else:
            logging.warning("  Warning: 'timestamp' column not found for time feature engineering.")

        np.random.seed(self.config["XGB_RANDOM_STATE"])
        basic_sensor_features = self.config["SENSOR_COLUMNS_FOR_OUTLIER_REMOVAL"]
        for feature in basic_sensor_features:
            if feature in df_new.columns and pd.api.types.is_numeric_dtype(df_new[feature]):
                noise = np.random.normal(0, df_new[feature].std() * noise_level, len(df_new))
                df_new[f'{feature}_noisy'] = df_new[feature] + noise
            else:
                logging.warning(f"  Warning: Sensor feature '{feature}' not found or not numeric for noise addition.")
        logging.info("  Time-based and noisy sensor features created.")
        return df_new