# realtime_machine_state_predict/model_inference.py

import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import Union, Tuple
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    Handles machine learning model loading and prediction operations.
    Encapsulates model interactions following Single Responsibility Principle.
    """
    
    def __init__(self, model_path: str, class_names_path: str):
        """
        Initialize predictor with model and class names file paths.
        
        Args:
            model_path (str): Path to the trained model file.
            class_names_path (str): Path to the class names text file.
        """
        self.model = None
        self.label_encoder = None
        self._load_model(model_path, class_names_path)

    def _load_model(self, model_path: str, class_names_path: str):
        """Load pre-trained model and label encoder from files."""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from '{model_path}'")
            
            if not os.path.exists(class_names_path):
                raise FileNotFoundError(f"Class names file not found: {class_names_path}")
            
            with open(class_names_path, 'r') as f:
                class_names = [line.strip() for line in f if line.strip()] # Filter out empty lines
            
            if not class_names:
                raise ValueError("Class names file is empty or contains no valid classes.")

            self.label_encoder = LabelEncoder()
            # Fit label encoder with the classes found in the file
            self.label_encoder.fit(class_names)
            logger.info(f"LabelEncoder loaded successfully. Classes: {self.label_encoder.classes_.tolist()}")
            
        except FileNotFoundError as e:
            logger.error(f"Required file not found: {e}. Model could not be loaded. Prediction system will not start.")
            self.model = None
            self.label_encoder = None
        except ValueError as e:
            logger.error(f"Data error in class names file: {e}. Model could not be loaded. Prediction system will not start.")
            self.model = None
            self.label_encoder = None
        except Exception as e:
            logger.error(f"Unexpected error loading model components: {e}", exc_info=True)
            logger.error("Prediction system will not start.")
            self.model = None
            self.label_encoder = None

    def predict(self, preprocessed_df: pd.DataFrame) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        """
        Perform predictions using the loaded model.
        
        Args:
            preprocessed_df (pd.DataFrame): Preprocessed DataFrame with features, in the correct order.
            
        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]: 
                Tuple of (predicted_labels, probabilities) or (None, None) if prediction fails.
        """
        if self.model is None or self.label_encoder is None:
            logger.error("Model or LabelEncoder not loaded. Cannot make predictions.")
            return None, None
        
        if preprocessed_df.empty:
            logger.warning("Received empty DataFrame for prediction.")
            return None, None

        try:
            probabilities = self.model.predict_proba(preprocessed_df)
            # Inverse transform the predicted class indices back to original class names
            predicted_labels = self.label_encoder.inverse_transform(np.argmax(probabilities, axis=1))
            return predicted_labels, probabilities
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return None, None

class PredictionResultProcessor:
    """
    Combines raw predictions with original data into comprehensive result DataFrame.
    Applies business logic such as error thresholding and result enrichment.
    """
    
    def __init__(self, label_encoder: LabelEncoder, error_threshold: float):
        """
        Initialize result processor with label encoder and error threshold.
        
        Args:
            label_encoder (LabelEncoder): Fitted label encoder for class names.
            error_threshold (float): Threshold for overriding predictions to 'error' class.
        """
        self.label_encoder = label_encoder
        self.error_threshold = error_threshold
        
        self.error_class_idx = -1
        try:
            if 'error' in self.label_encoder.classes_:
                self.error_class_idx = list(self.label_encoder.classes_).index('error')
            else:
                logger.warning("'error' class not found in LabelEncoder classes. Error-specific logic will be skipped.")
        except ValueError: # Should not happen if 'error' is in classes, but defensive
             logger.warning("'error' class not found in LabelEncoder classes during index lookup. Error-specific logic will be skipped.")


    def process(self, original_df: pd.DataFrame, predicted_labels: np.ndarray, probabilities: np.ndarray) -> Union[pd.DataFrame, None]:
        """
        Combine original data with model predictions and apply business rules.
        
        Args:
            original_df (pd.DataFrame): Original raw input DataFrame.
            predicted_labels (np.ndarray): Predicted labels from model.
            probabilities (np.ndarray): Probability scores for all classes.
            
        Returns:
            Union[pd.DataFrame, None]: Enhanced DataFrame with probabilities and overrides or None if processing fails.
        """
        if original_df.empty or predicted_labels is None or probabilities is None:
            logger.warning("Invalid input received for result processing (empty data or null predictions/probabilities).")
            return None

        # Select essential columns from original DataFrame
        info_columns = [col for col in ['machine_id', 'timestamp', 'product_type', 'temperature', 
                                        'pressure', 'energy_consumption', 'vibration', 'actual_state'] 
                        if col in original_df.columns]
        
        result_df = original_df[info_columns].copy()
        
        # Ensure predicted_labels match original_df's index if they differ
        if len(predicted_labels) != len(result_df):
            logger.error(f"Mismatch in prediction count ({len(predicted_labels)}) and original data rows ({len(result_df)}). Cannot process results.")
            return None

        result_df['predicted_state'] = predicted_labels
        
        # Add probability columns for each class
        class_names = self.label_encoder.classes_
        for i, class_name in enumerate(class_names):
            if i < probabilities.shape[1]: # Defensive check against shape mismatch
                result_df[f'prob_{class_name}'] = probabilities[:, i]
            else:
                result_df[f'prob_{class_name}'] = 0.0 # Assign 0 if probability column missing (unlikely if model works)
                logger.warning(f"Probability column for '{class_name}' missing from model output. Assigned 0.0.")
        
        # Handle error probability and threshold-based overrides
        result_df['error_probability'] = 0.0 # Initialize
        if self.error_class_idx != -1:
            if self.error_class_idx < probabilities.shape[1]:
                error_probabilities = probabilities[:, self.error_class_idx]
                result_df['error_probability'] = error_probabilities
                
                # Override predictions exceeding error threshold to 'error' state
                override_mask = (error_probabilities > self.error_threshold)
                num_overrides = override_mask.sum()
                if num_overrides > 0:
                    result_df.loc[override_mask, 'predicted_state'] = 'error'
                    logger.info(f"Overridden {num_overrides} predictions to 'error' based on probability threshold {self.error_threshold:.4f}.")
            else:
                logger.warning(f"Error class index {self.error_class_idx} out of bounds for probabilities array. Error overriding skipped.")
        else:
            logger.info("Error class not present in model classes. No error-specific overriding applied.")

        # Add metadata columns
        result_df['prediction_time'] = datetime.now()
        result_df['prediction_mode'] = "REALTIME_MODEL" # Clarify mode
        
        # Add error codes for error states
        result_df['error_code'] = 'N/A'
        result_df.loc[result_df['predicted_state'] == 'error', 'error_code'] = 'High Risk Anomaly Detected'
        
        return result_df