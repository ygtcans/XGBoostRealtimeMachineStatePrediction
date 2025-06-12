# xgboost_model/data_splitting_encoding.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import logging

class DataSplitterEncoder:
    """
    Manages data splitting, outlier removal on splits, and categorical encoding.
    """
    def __init__(self, config: dict):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = None
        self.all_product_types = []
        self.ohe_feature_names = []

    def prepare_data_for_modeling(self, df: pd.DataFrame, data_processor) -> tuple:
        # ... (Mevcut prepare_data_for_modeling metodunun içeriği) ...
        logging.info("\n--- Data Splitting and Feature Engineering ---")

        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must contain a 'timestamp' column for time-based splitting.")
        if 'state' not in df.columns:
            raise ValueError("DataFrame must contain a 'state' column as target variable.")

        df_sorted = df.sort_values('timestamp').copy()
        split_time = df_sorted['timestamp'].quantile(self.config["TRAIN_TEST_SPLIT_QUANTILE"])

        train_df = df_sorted[df_sorted['timestamp'] <= split_time].copy()
        test_df = df_sorted[df_sorted['timestamp'] > split_time].copy()

        test_df_original_copy = test_df.copy()

        logging.info(f"Time-based split: Train: {train_df.shape}, Test: {test_df.shape}")

        train_df = data_processor.remove_outliers_iqr(train_df, self.config["SENSOR_COLUMNS_FOR_OUTLIER_REMOVAL"])
        test_df = data_processor.remove_outliers_iqr(test_df, self.config["SENSOR_COLUMNS_FOR_OUTLIER_REMOVAL"])

        logging.info(f"After outlier removal: Train size: {train_df.shape}, Test size: {test_df.shape}")

        logging.info("\nApplying feature engineering to train and test sets...")
        train_df = data_processor.create_time_and_noisy_features(train_df)
        test_df = data_processor.create_time_and_noisy_features(test_df)

        self._apply_one_hot_encoding(train_df, test_df, df)

        logging.info("\nPreparing target and features...")
        unique_states = sorted(df['state'].dropna().unique())
        self.label_encoder.fit(unique_states)
        class_names = list(self.label_encoder.classes_)
        logging.info(f"Classes found and encoded: {class_names}")

        train_df['state_encoded'] = self.label_encoder.transform(train_df['state'])
        test_df['state_encoded'] = self.label_encoder.transform(test_df['state'])

        features_to_keep_base = [
            'temperature', 'pressure', 'energy_consumption', 'vibration',
            'hour_of_day', 'is_weekend',
            'temperature_noisy', 'pressure_noisy', 'energy_consumption_noisy', 'vibration_noisy'
        ]

        features_to_keep = [f for f in features_to_keep_base if f in train_df.columns and f in test_df.columns] + self.ohe_feature_names

        final_features = []
        for feature in features_to_keep:
            if feature in train_df.columns and feature in test_df.columns:
                final_features.append(feature)
            else:
                logging.warning(f"  Warning: Feature '{feature}' not found in both train and test sets after transformations. It will be excluded.")
        features_to_keep = final_features

        X_train = train_df[features_to_keep]
        X_test = test_df[features_to_keep]
        y_train = train_df['state_encoded']
        y_test = test_df['state_encoded']

        logging.info(f"Number of features used for training: {len(features_to_keep)}")
        logging.info(f"Class distribution (train): {train_df['state'].value_counts().to_dict()}")
        logging.info(f"Class distribution (test): {test_df['state'].value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test, features_to_keep, class_names, self.label_encoder, test_df_original_copy

    def _apply_one_hot_encoding(self, train_df: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame):
        # ... (Mevcut _apply_one_hot_encoding metodunun içeriği) ...
        logging.info("\nApplying One-Hot Encoding for 'product_type'...")

        if 'product_type' in train_df.columns and train_df['product_type'].dtype == 'object':
            self.all_product_types = sorted(full_df['product_type'].dropna().unique())

            self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[self.all_product_types])

            ohe_train_array = self.one_hot_encoder.fit_transform(train_df[['product_type']])
            ohe_test_array = self.one_hot_encoder.transform(test_df[['product_type']])

            self.ohe_feature_names = list(self.one_hot_encoder.get_feature_names_out(['product_type']))

            train_df.drop(columns=['product_type'], inplace=True)
            test_df.drop(columns=['product_type'], inplace=True)

            train_df[self.ohe_feature_names] = ohe_train_array
            test_df[self.ohe_feature_names] = ohe_test_array

            logging.info(f"  'product_type' column successfully One-Hot Encoded. Added {len(self.ohe_feature_names)} new features.")
            logging.info(f"  Train columns after OHE: {train_df.shape[1]}, Test columns after OHE: {test_df.shape[1]}")
        else:
            logging.warning("  'product_type' column not found or not suitable for One-Hot Encoding. Skipping OHE.")
            self.ohe_feature_names = []