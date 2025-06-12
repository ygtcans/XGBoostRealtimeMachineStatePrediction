# xgboost_model/main.py
import os
import joblib
import logging
import warnings

from config import CONFIG
from data_processing import DataProcessor
from data_splitting_encoding import DataSplitterEncoder
from model_training_evaluation import ModelTrainer

# Configure logging and warnings
logging.basicConfig(level=CONFIG["LOGGING_LEVEL"], format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore') # Suppress warnings from libraries for cleaner output

def main_training_workflow(config: dict):
    """
    Orchestrates the entire machine learning workflow: data loading, preprocessing,
    splitting, feature engineering, model training, evaluation, and saving.
    """
    logging.info("\n--- Machine State Prediction Model Training Workflow Started ---")

    data_processor = DataProcessor(config)
    data_splitter_encoder = DataSplitterEncoder(config)
    model_trainer = ModelTrainer(config)

    try:
        df = data_processor.load_and_clean_data(config["DATA_FILE_PATH"])

        X_train, X_test, y_train, y_test, features_to_keep, class_names, label_encoder, test_df_original = \
            data_splitter_encoder.prepare_data_for_modeling(df, data_processor)

        trained_model = model_trainer.train_xgboost_model(X_train, y_train, features_to_keep)

        y_pred, y_pred_proba = model_trainer.evaluate_model_metrics(X_train, y_train, X_test, y_test, class_names)

        model_trainer.analyze_model_diagnosis(X_test, y_test, y_pred, class_names, features_to_keep, test_df_original)

        model_trainer.compare_with_baselines(X_train, y_train, X_test, y_test, y_pred)

        model_trainer.cross_validation_check(X_train, y_train, y_test, y_pred)

        model_trainer.improve_error_class_performance(y_test, y_pred_proba, label_encoder, class_names)

        # --- Model Saving ---
        logging.info("\n--- Saving Model and Artifacts ---")
        os.makedirs(config["MODEL_SAVE_DIRECTORY"], exist_ok=True)

        model_save_path = os.path.join(config["MODEL_SAVE_DIRECTORY"], config["MODEL_FILENAME"])
        joblib.dump(trained_model, model_save_path)
        logging.info(f"Main XGBoost model saved to '{model_save_path}'.")

        class_names_path = os.path.join(config["MODEL_SAVE_DIRECTORY"], config["CLASS_NAMES_FILENAME"])
        with open(class_names_path, 'w') as f:
            for item in class_names:
                f.write("%s\n" % item)
        logging.info(f"Class names saved to '{class_names_path}'.")

        product_types_path = os.path.join(config["MODEL_SAVE_DIRECTORY"], config["PRODUCT_TYPES_FILENAME"])
        with open(product_types_path, 'w') as f:
            for item in data_splitter_encoder.all_product_types:
                f.write("%s\n" % item)
        logging.info(f"Product types used for OHE saved to '{product_types_path}'.")

        features_path = os.path.join(config["MODEL_SAVE_DIRECTORY"], config["FEATURES_FILENAME"])
        with open(features_path, 'w') as f:
            for item in features_to_keep:
                f.write(f"{item}\n")
        logging.info(f"Feature names saved to '{features_path}'.")

    except Exception as e:
        logging.critical(f"An unrecoverable error occurred during the workflow: {e}", exc_info=True)
        print(f"Workflow terminated due to a critical error: {e}")
    finally:
        logging.info("\n--- Machine State Prediction Model Training Workflow Completed ---")


if __name__ == "__main__":
    main_training_workflow(CONFIG)