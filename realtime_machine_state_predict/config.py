# realtime_machine_state_predict/config.py

import logging

# Configure logging to display INFO, WARNING, ERROR, and CRITICAL messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
CONFIG = {
    # File paths for model and auxiliary files
    "MODEL_PATH": 'model/xgboost_model.pkl',
    "CLASS_NAMES_PATH": 'model/class_names.txt',
    "PRODUCT_TYPES_PATH": 'model/product_types_for_ohe.txt',
    "FEATURES_TO_KEEP_PATH": 'model/features_to_keep.txt',
    
    # Optimized threshold for 'error' class classification
    "OPTIMAL_ERROR_THRESHOLD": 0.2567, # This value should ideally come from training evaluation

    # WebSocket URL with embedded API key for sensor data (IMPORTANT: Replace with your actual key in production)
    "WS_URL": "wss://yigitcandursun.com/ws/sensordata?api_key=e3f4b2a5c77d4a01b3e8d8f9a9c0f1d2",
    
    # System monitoring and connection settings
    "STATS_UPDATE_INTERVAL_SECONDS": 60,
    "WEBSOCKET_RETRY_DELAY_SECONDS": 5,
    "WEBSOCKET_MAX_RETRIES": 5,
    "PREDICTION_HISTORY_MAX_SIZE": 100,
    "HIGH_RISK_PROBABILITY_THRESHOLD": 0.3
}