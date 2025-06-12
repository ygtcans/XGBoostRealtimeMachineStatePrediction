# realtime_machine_state_predict/system_orchestrator.py

import pandas as pd
import numpy as np
import asyncio
import websockets
from datetime import datetime
import signal
import os
from typing import Union, List

# Import modules from our new package structure
from config import CONFIG, logger
from data_processing import WebSocketDataConverter, DataPreprocessor
from model_inference import ModelPredictor, PredictionResultProcessor
from reporting import SystemReporter

class RealTimePredictionSystem:
    """
    Main orchestrator for the real-time machine state prediction process.
    Implements Facade pattern as the primary coordinator for all subsystems.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the real-time prediction system with configuration.
        
        Args:
            config (dict): Configuration dictionary containing all system settings
        """
        self.config = config
        self.running = True
        self.prediction_history = []
        self.stats = {
            'total_predictions': 0,
            'error_predictions': 0,
            'high_risk_predictions': 0,
            'start_time': datetime.now()
        }

        # --- Load auxiliary files for robust initialization ---
        # 1. Load class names for LabelEncoder
        self.model_predictor = ModelPredictor(self.config["MODEL_PATH"], self.config["CLASS_NAMES_PATH"])
        if not self.model_predictor.model or not self.model_predictor.label_encoder:
            logger.critical("Core prediction components (model/label encoder) could not be loaded. Stopping system initialization.")
            self.running = False
            return

        # 2. Load product types for OneHotEncoder
        product_types = self._load_auxiliary_file(self.config["PRODUCT_TYPES_PATH"], "product types")
        if not product_types:
            logger.critical("Product types could not be loaded. Stopping system initialization.")
            self.running = False
            return

        # 3. Load features_to_keep for consistent preprocessing
        features_expected = self._load_auxiliary_file(self.config["FEATURES_TO_KEEP_PATH"], "expected features")
        if not features_expected:
            logger.critical("Expected features list could not be loaded. Stopping system initialization.")
            self.running = False
            return

        # Initialize helper classes using Dependency Injection principles
        self.ws_converter = WebSocketDataConverter()
        self.data_preprocessor = DataPreprocessor(product_types, features_expected)
        self.result_processor = PredictionResultProcessor(
            self.model_predictor.label_encoder, 
            self.config["OPTIMAL_ERROR_THRESHOLD"]
        )
        self.reporter = SystemReporter(
            self.model_predictor.label_encoder, 
            self.config["HIGH_RISK_PROBABILITY_THRESHOLD"]
        )
        
        # Set up OS signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _load_auxiliary_file(self, path: str, file_type: str) -> Union[List[str], None]:
        """
        Generic method to load a list of strings from a text file.
        
        Args:
            path (str): Path to the text file.
            file_type (str): Description of the file's content (e.g., "product types", "expected features").
            
        Returns:
            Union[List[str], None]: List of strings or None if loading fails.
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{file_type.capitalize()} file not found: {path}")
            
            with open(path, 'r') as f:
                items = [line.strip() for line in f if line.strip()] # Filter out empty lines
            
            if not items:
                raise ValueError(f"{file_type.capitalize()} file is empty or contains no valid entries: {path}")
            
            logger.info(f"{file_type.capitalize()} loaded successfully (count: {len(items)}).")
            return items
        except FileNotFoundError as e:
            logger.error(f"Error loading {file_type}: {e}.")
            return None
        except ValueError as e:
            logger.error(f"Data error in {file_type} file: {e}.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {file_type}: {e}", exc_info=True)
            return None

    def _setup_signal_handlers(self):
        """Configure signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Signal handler callback to stop the system gracefully."""
        logger.info(f"Shutdown signal received (Signal: {signum}). Initiating graceful shutdown...")
        self.running = False

    async def _periodic_stats_async(self):
        """Display aggregate system statistics at regular intervals."""
        while self.running:
            await asyncio.sleep(self.config["STATS_UPDATE_INTERVAL_SECONDS"])
            if self.running: # Check again in case shutdown signal arrived during sleep
                self.reporter.display_system_stats(self.stats)

    async def _process_websocket_message(self, message: str):
        """
        Process a single WebSocket message from data conversion to display.
        
        Args:
            message (str): Raw WebSocket message string.
        """
        try:
            # Convert raw data to DataFrame
            df_raw = self.ws_converter.convert(message)
            if df_raw is None or df_raw.empty:
                logger.warning("No valid raw data received for processing.")
                return

            # Preprocess DataFrame for model
            preprocessed_df = self.data_preprocessor.preprocess(df_raw)
            if preprocessed_df.empty:
                logger.warning("Preprocessing resulted in empty DataFrame. Skipping prediction.")
                return

            # Get predictions from model
            predicted_labels, probabilities = self.model_predictor.predict(preprocessed_df)
            if predicted_labels is None or probabilities is None:
                logger.warning("Prediction failed. Skipping result processing.")
                return

            # Process and enrich prediction results
            result_df = self.result_processor.process(df_raw, predicted_labels, probabilities)
            if result_df is None:
                logger.warning("Result processing failed. Skipping display.")
                return

            # Display real-time predictions to console
            self.reporter.display_realtime_predictions(result_df)
            
            # Update system statistics
            self.stats['total_predictions'] += len(result_df)
            self.stats['error_predictions'] += (result_df['predicted_state'] == 'error').sum()
            self.stats['high_risk_predictions'] += (result_df['error_probability'] > self.config["HIGH_RISK_PROBABILITY_THRESHOLD"]).sum()
            
            # Maintain prediction history in memory with size limit
            self.prediction_history.append(result_df)
            if len(self.prediction_history) > self.config["PREDICTION_HISTORY_MAX_SIZE"]:
                self.prediction_history.pop(0) # Remove oldest entry

        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}", exc_info=True)

    async def connect_and_predict(self):
        """Establish WebSocket connection and manage real-time prediction loop."""
        logger.info(f"Attempting to connect to WebSocket: {self.config['WS_URL']}.")
        
        retry_count = 0
        
        # Attempt connection up to maximum retry limit
        while self.running and retry_count < self.config["WEBSOCKET_MAX_RETRIES"]:
            try:
                # Establish WebSocket connection with async context manager
                async with websockets.connect(
                    self.config["WS_URL"],
                    ping_interval=30,  # Keep-alive ping interval
                    ping_timeout=10,   # Timeout for ping response
                    close_timeout=10   # Timeout for connection close
                ) as websocket:
                    
                    logger.info("WebSocket connection established!")
                    logger.info("Starting real-time prediction stream...")
                    print("-" * 80) # Visual separator for console output
                    
                    retry_count = 0  # Reset retry counter on successful connection
                    
                    # Start periodic statistics as separate task
                    asyncio.create_task(self._periodic_stats_async())
                    
                    # Process each incoming message until connection closes or system stops
                    async for message in websocket:
                        if not self.running:
                            logger.info("System shutdown initiated. Breaking WebSocket message loop.")
                            break
                        await self._process_websocket_message(message)
                            
            except websockets.exceptions.ConnectionClosed as e:
                retry_count += 1
                logger.warning(f"WebSocket connection closed unexpectedly (Code: {e.code}, Reason: {e.reason}).")
                if self.running and retry_count < self.config["WEBSOCKET_MAX_RETRIES"]:
                    logger.info(f"Retrying connection in {self.config['WEBSOCKET_RETRY_DELAY_SECONDS']} seconds... (Attempt {retry_count}/{self.config['WEBSOCKET_MAX_RETRIES']})")
                    await asyncio.sleep(self.config["WEBSOCKET_RETRY_DELAY_SECONDS"])
                
            except websockets.exceptions.InvalidURI:
                logger.critical(f"Invalid WebSocket URL: {self.config['WS_URL']}. Please check the URL. Terminating system.")
                self.running = False # Critical error, no retries for invalid URI
                break
                
            except Exception as e:
                retry_count += 1
                logger.error(f"General WebSocket connection error: {e}", exc_info=True)
                if self.running and retry_count < self.config["WEBSOCKET_MAX_RETRIES"]:
                    logger.info(f"Retrying connection in {self.config['WEBSOCKET_RETRY_DELAY_SECONDS']} seconds... (Attempt {retry_count}/{self.config['WEBSOCKET_MAX_RETRIES']})")
                    await asyncio.sleep(self.config["WEBSOCKET_RETRY_DELAY_SECONDS"])
        
        # Log critical error if maximum retries reached and system is still supposed to be running
        if retry_count >= self.config["WEBSOCKET_MAX_RETRIES"] and self.running:
            logger.critical(f"Maximum retry attempts ({self.config['WEBSOCKET_MAX_RETRIES']}) reached. WebSocket connection could not be established. Terminating system.")
            self.running = False
        else:
            logger.info("WebSocket connection loop finished (graceful shutdown or maximum retries reached).")

    async def run(self):
        """Start the main real-time prediction system."""
        try:
            # Check if critical components loaded successfully before starting connection loop
            if (self.running and self.model_predictor.model and self.model_predictor.label_encoder 
                and self.data_preprocessor and self.ws_converter and self.result_processor and self.reporter):
                logger.info("All system components initialized successfully. Starting prediction workflow.")
                await self.connect_and_predict()
            else:
                logger.critical("Prediction system cannot start due to critical initialization errors. Check logs for details.")
        except asyncio.CancelledError:
            logger.info("Asyncio task cancelled. System shutting down.")
        except KeyboardInterrupt:
            logger.info("System stopped by user (KeyboardInterrupt).")
        except Exception as e:
            logger.critical(f"An unexpected error occurred while system was running: {e}", exc_info=True)
        finally:
            self.running = False # Ensure running flag is false on exit
            # Give a small moment for pending tasks to finish logging
            await asyncio.sleep(0.5) 
            self.reporter.display_system_stats(self.stats)
            logger.info("System shut down gracefully.")