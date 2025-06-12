# xgboost_model/config.py
import logging

CONFIG = {
    # File paths
    "DATA_FILE_PATH": 'data/raw_sensor_data_20250604_144909.csv',
    "MODEL_SAVE_DIRECTORY": 'model',
    "MODEL_FILENAME": 'xgboost_model.pkl',
    "CLASS_NAMES_FILENAME": 'class_names.txt',
    "PRODUCT_TYPES_FILENAME": 'product_types_for_ohe.txt',
    "FEATURES_FILENAME": 'features_to_keep.txt',

    # Data Preprocessing
    "DATETIME_COLUMNS": ['timestamp', 'fetched_at'],
    "SENSOR_COLUMNS_FOR_OUTLIER_REMOVAL": ['temperature', 'pressure', 'energy_consumption', 'vibration'],
    "IQR_OUTLIER_FACTOR": 1.5,
    "NOISE_LEVEL_FOR_FEATURES": 0.01,
    "TRAIN_TEST_SPLIT_QUANTILE": 0.7,

    # Model Parameters (XGBoost)
    "XGB_N_ESTIMATORS": 150,
    "XGB_LEARNING_RATE": 0.2,
    "XGB_MAX_DEPTH": 3,
    "XGB_MIN_CHILD_WEIGHT": 5,
    "XGB_SUBSAMPLE": 0.7,
    "XGB_COLSAMPLE_BYTREE": 0.7,
    "XGB_REG_ALPHA": 0.1,
    "XGB_REG_LAMBDA": 0.1,
    "XGB_RANDOM_STATE": 42,
    "XGB_OBJECTIVE": 'multi:softprob',
    "XGB_EVAL_METRIC": 'mlogloss',

    # Evaluation Thresholds
    "HIGH_ACCURACY_THRESHOLD": 0.95,
    "HIGH_DISCRIMINATIVE_POWER_RATIO": 5,
    "DOMINANT_FEATURE_IMPORTANCE_THRESHOLD": 0.5,
    "CV_TEST_PERFORMANCE_GAP_THRESHOLD": 0.05,
    "ERROR_CLASS_IMBALANCE_THRESHOLD": 0.10,

    # Logging
    "LOGGING_LEVEL": logging.INFO
}