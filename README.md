# XGBoost Realtime Machine State Prediction

---

## 🚀 Overview

This repository houses a comprehensive machine learning project designed for **real-time prediction of machine states** using an **XGBoost Classifier**. The system aims to monitor industrial machinery by processing incoming sensor data, predicting its operational state (e.g., `normal`, `warning`, `error`), and enabling proactive interventions for improved maintenance and operational efficiency.

The project is structured into two main, interconnected components:

1.  **Training Workflow (`xgboost_model/`):** Contains the full machine learning model lifecycle, including data loading, preprocessing, feature engineering, model training, rigorous evaluation, and persistence of the trained model and artifacts.
2.  **Real-Time Prediction System (`realtime_machine_state_predict/`):** An asynchronous system that connects to a WebSocket, ingests live sensor data, processes it, performs real-time state predictions using the pre-trained XGBoost model, and provides immediate reporting and alerts.

---

## ✨ Features

### Model Training & Evaluation Workflow (`xgboost_model/`)

* **Robust Data Processing:**
    * Loads and cleans CSV data, standardizing column names, converting datetime fields, handling NaNs, and removing duplicate rows.
    * Applies IQR-based outlier removal to specified sensor features.
    * Creates time-based features (`hour_of_day`, `is_weekend`) and adds Gaussian noise to sensor features for robustness testing.
* **Time-Based Data Splitting:** Divides the dataset into training and testing sets based on a time quantile, ensuring realistic evaluation for time-series data.
* **Comprehensive Categorical Encoding:** Uses `LabelEncoder` for the target `state` variable and `OneHotEncoder` for `product_type`, ensuring consistent encoding across datasets.
* **XGBoost Model Training:** Implements a scalable XGBoost Classifier, integrating a `StandardScaler` within a `Pipeline` for numerical feature preprocessing.
* **In-depth Model Evaluation:**
    * Calculates and reports standard classification metrics (`accuracy`, `precision`, `recall`, `f1-score`) and displays a confusion matrix.
    * Performs diagnostic analysis to identify misclassified instances.
    * Compares XGBoost performance against baseline models (e.g., `DummyClassifier`).
    * Converts `DataFrame` objects in diagnostic analysis to string before logging to prevent potential issues.
    * Conducts time-series cross-validation to assess model stability and generalization.
* **Critical Error Class Performance:** Dedicated analysis for the "error" state, including:
    * Class imbalance assessment in the test set.
    * Optimal probability threshold determination using Precision-Recall curves to maximize F1-score for error detection.
* **Model Persistence:** Saves the trained XGBoost model, `LabelEncoder` classes, `OneHotEncoder` product types, and feature names to disk for later use in the real-time prediction system.

### Real-Time Prediction System (`realtime_machine_state_predict/`)

* **WebSocket Data Ingestion:** Connects to a WebSocket endpoint (`WS_URL` from `realtime_machine_state_predict/config.py`) to receive live sensor data in JSON format.
* **Asynchronous Processing:** Leverages `asyncio` for efficient handling of concurrent WebSocket communication and prediction tasks.
* **Dynamic Data Preprocessing:** Transforms raw JSON sensor data into a structured format, applies necessary preprocessing steps (e.g., datetime conversion, feature creation, One-Hot Encoding) to align with the trained model's input requirements.
* **Live Inference:** Uses the pre-trained XGBoost model to make real-time predictions on incoming sensor data.
* **Error State Prioritization:** Applies an `OPTIMAL_ERROR_THRESHOLD` (loaded from `realtime_machine_state_predict/config.py`) to override predictions to 'error' if the predicted error probability exceeds this critical threshold, enhancing anomaly detection.
* **Interactive Reporting:** Displays real-time predictions with color-coding for different machine states (`normal`, `warning`, `error`, `high_risk`) directly in the console.
* **System Health Monitoring:** Provides aggregate statistics on total predictions, error predictions, high-risk predictions, and system runtime.
* **Robustness:** Implements retry mechanisms for WebSocket connection failures.

---

## 🛠️ Getting Started

Follow these instructions to set up the project and run both the training workflow and the real-time prediction system on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ygtcans/XGBoostRealtimeMachineStatePrediction.git
    cd XGBoostRealtimeMachineStatePrediction
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` does not exist, create it by running `pip freeze > requirements.txt` after manually installing `pandas`, `numpy`, `scikit-learn`, `xgboost`, `websockets`, `joblib`.)*

### Project Structure
   ```bash
      XGBoostRealtimeMachineStatePrediction/
    ├── data/                                   # Contains raw input data files
    │   └── raw_sensor_data_20250604_144909.csv
    ├── model/                                  # Stores trained model and auxiliary artifacts
    │   ├── class_names.txt
    │   ├── features_to_keep.txt
    │   ├── product_types_for_ohe.txt
    │   └── xgboost_model.pkl
    ├── realtime_machine_state_predict/         # Contains the Real-Time Prediction System's code
    │   ├── init.py
    │   ├── config.py                           # Configuration for the real-time system
    │   ├── data_processing.py                  # Data processing for real-time WebSocket data
    │   ├── main.py                             # Entry point for the real-time prediction system
    │   ├── model_inference.py                  # Model loading and real-time prediction logic
    │   ├── reporting.py                        # Real-time prediction display and system stats
    │   └── system_orchestrator.py              # Main orchestrator for the real-time system
    ├── requirements.txt                        # Python package dependencies
    ├── venv/                                   # Python virtual environment folder
    └── xgboost_model/                          # Contains the Model Training & Evaluation Workflow's code
    ├── config.py                           # Configuration for the training workflow
    ├── data_processing.py                  # Data loading, cleaning, feature creation for training
    ├── data_splitting_encoding.py          # Data splitting, outlier handling, categorical encoding
    ├── main.py                             # Entry point for the TRAINING workflow
    └── model_training_evaluation.py        # Model training, evaluation, and analysis
   ```

---

## 🏃 Usage

### A. Training the Machine State Prediction Model

To train the XGBoost model and save its artifacts:

1.  **Prepare your dataset:** Ensure your historical machine sensor data (e.g., `raw_sensor_data_20250604_144909.csv`) is placed in the `data/` directory.

2.  **Configure `xgboost_model/config.py`:**
    * Update `CONFIG["DATA_FILE_PATH"]` within `xgboost_model/config.py` to point to your dataset (e.g., `'../data/raw_sensor_data_20250604_144909.csv'`).
    * Adjust other parameters like `DATETIME_COLUMNS`, `SENSOR_COLUMNS_FOR_OUTLIER_REMOVAL`, `TRAIN_TEST_SPLIT_QUANTILE`, `XGB_PARAMS`, and `ERROR_CLASS_IMBALANCE_THRESHOLD` as needed for your specific data and model requirements.
    * Ensure `MODEL_SAVE_DIRECTORY`, `MODEL_FILENAME`, `CLASS_NAMES_FILENAME`, `PRODUCT_TYPES_FILENAME`, `FEATURES_FILENAME` are correctly set for saving artifacts (e.g., `'../model/'`).

3.  **Run the training workflow:**
    ```bash
    python xgboost_model/main.py
    ```
    Upon successful completion, the trained model (`xgboost_model.pkl`), class names, product types for OHE, and feature names will be saved in the `model/` directory.

### B. Running the Real-Time Machine State Prediction System

To run the system that connects to a WebSocket and makes live predictions:

1.  **Ensure Model Artifacts Exist:** Make sure you have already run the training workflow (Section A) and that the `model/` directory contains:
    * `xgboost_model.pkl`
    * `class_names.txt`
    * `product_types_for_ohe.txt`
    * `features_to_keep.txt`

2.  **Configure `realtime_machine_state_predict/config.py`:**
    * Set `CONFIG["MODEL_PATH"]`, `CONFIG["CLASS_NAMES_PATH"]`, `CONFIG["PRODUCT_TYPES_PATH"]`, `CONFIG["FEATURES_TO_KEEP_PATH"]` to the correct paths of your saved artifacts (e.g., `'../model/xgboost_model.pkl'`).
    * **Crucially, update `CONFIG["WS_URL"]` with your actual WebSocket sensor data endpoint.** (e.g., `wss://your-data-source.com/ws/sensordata?api_key=YOUR_API_KEY`).
    * Adjust `OPTIMAL_ERROR_THRESHOLD` (this value is loaded from `config.py` in the real-time system), `HIGH_RISK_PROBABILITY_THRESHOLD`, `WEBSOCKET_RETRY_DELAY_SECONDS`, etc., as per your real-time operational needs.

3.  **Execute the real-time system:**
    ```bash
    python realtime_machine_state_predict/main.py
    ```
    The system will start connecting to the WebSocket, process incoming data, display real-time predictions, and provide system statistics.

---

## ⚙️ Configuration

It's important to note that **this project uses two separate `config.py` files**:

* **`xgboost_model/config.py`**: Specifically for the **training workflow**. Defines parameters related to data paths for training, model hyperparameters, and saving locations for training artifacts.
* **`realtime_machine_state_predict/config.py`**: Specifically for the **real-time prediction system**. Defines parameters related to loading the trained model, WebSocket connection details, real-time thresholds (e.g., `OPTIMAL_ERROR_THRESHOLD`), and system monitoring settings.

Please ensure you configure the correct `config.py` file for the workflow you intend to run.

Key parameters you might need to adjust in these files include:

* **File Paths:**
    * `DATA_FILE_PATH` (in `xgboost_model/config.py`): Path to the raw dataset used for training.
    * `MODEL_SAVE_DIRECTORY` (in `xgboost_model/config.py`): Directory to save trained model artifacts (e.g., `'../model/'`).
    * `MODEL_PATH`, `CLASS_NAMES_PATH`, `PRODUCT_TYPES_PATH`, `FEATURES_TO_KEEP_PATH` (in `realtime_machine_state_predict/config.py`): Full paths to load the trained model and its dependencies for inference.
* **Data Processing:**
    * `DATETIME_COLUMNS`: List of columns to convert to datetime.
    * `SENSOR_COLUMNS_FOR_OUTLIER_REMOVAL`: Numeric columns for outlier detection.
    * `NOISE_LEVEL_FOR_FEATURES`: Factor for adding noise to sensor features during training.
* **Training & Evaluation:**
    * `TRAIN_TEST_SPLIT_QUANTILE`: Defines the split point for time-based train/test sets.
    * `XGB_RANDOM_STATE`: Seed for reproducibility.
    * `XGB_PARAMS`: Dictionary of XGBoost hyperparameters.
    * `ERROR_CLASS_IMBALANCE_THRESHOLD`: Threshold to flag significant imbalance in the 'error' class.
* **Real-Time Specific:**
    * `WS_URL`: WebSocket URL for live sensor data. **(Critical: Replace placeholder with your actual endpoint!)**
    * `OPTIMAL_ERROR_THRESHOLD`: Probability threshold for classifying a state as 'error' in real-time.
    * `HIGH_RISK_PROBABILITY_THRESHOLD`: Probability threshold for 'high risk' alerts (distinct from error classification).
    * `WEBSOCKET_RETRY_DELAY_SECONDS`, `WEBSOCKET_MAX_RETRIES`: WebSocket connection robustness settings.
    * `PREDICTION_HISTORY_MAX_SIZE`: Max number of predictions to keep in history.
    * `STATS_UPDATE_INTERVAL_SECONDS`: Frequency of system statistics display.
* **Logging:**
    * `LOGGING_LEVEL`: Controls the verbosity of log messages (e.g., `logging.INFO`, `logging.DEBUG`).

---
