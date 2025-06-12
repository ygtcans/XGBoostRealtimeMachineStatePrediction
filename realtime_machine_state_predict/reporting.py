# realtime_machine_state_predict/reporting.py

import pandas as pd
from datetime import datetime
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class SystemReporter:
    """
    Manages display of real-time predictions and aggregate system statistics.
    Focuses on output generation following Single Responsibility Principle.
    """
    
    def __init__(self, label_encoder: LabelEncoder, high_risk_threshold: float):
        """
        Initialize reporter with label encoder and high risk threshold.
        
        Args:
            label_encoder (LabelEncoder): Label encoder for class names.
            high_risk_threshold (float): Threshold for high risk probability alerts.
        """
        self.label_encoder = label_encoder
        self.high_risk_threshold = high_risk_threshold
        self.class_names = self.label_encoder.classes_
    
    def display_realtime_predictions(self, result_df: pd.DataFrame):
        """
        Display detailed real-time predictions for each machine with color coding.
        
        Args:
            result_df (pd.DataFrame): Processed prediction results.
        """
        if result_df is None or result_df.empty:
            return
            
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Calculate summary statistics
        error_count = (result_df['predicted_state'] == 'error').sum()
        high_risk_count = (result_df['error_probability'] > self.high_risk_threshold).sum()
        
        # Clear screen for a cleaner real-time display

        print(f"\n--- REAL-TIME PREDICTIONS ({timestamp}) ---")
        print(f"📊 Machines Processed: {len(result_df)} | 🚨 Predicted Errors: {error_count} | ⚠️  High Risk Machines: {high_risk_count}")
        print("-" * 80)
        
        # ANSI color codes for console output
        COLOR_RED = "\033[91m"
        COLOR_YELLOW = "\033[93m"
        COLOR_GREEN = "\033[92m"
        COLOR_BLUE = "\033[94m"
        COLOR_RESET = "\033[0m"

        # Order classes with 'error' first for better visibility if it exists
        ordered_classes = ['error'] + [c for c in self.class_names if c != 'error']

        # Display detailed information for each machine
        for _, machine in result_df.iterrows():
            machine_id = machine.get('machine_id', 'N/A')
            predicted_state = machine.get('predicted_state', 'N/A')
            actual_state = machine.get('actual_state', 'N/A') # Now directly from result_df
            product_type = machine.get('product_type', 'N/A')
            error_code = machine.get('error_code', 'N/A')
            error_prob = machine.get('error_probability', 0)
            
            # Extract sensor readings
            temp = machine.get('temperature', 'N/A')
            pressure = machine.get('pressure', 'N/A')
            energy = machine.get('energy_consumption', 'N/A')
            vibration = machine.get('vibration', 'N/A')

            # Determine color based on machine state
            status_color = COLOR_BLUE # Default for unknown/other states
            if predicted_state == 'error':
                status_color = COLOR_RED
            elif error_prob > self.high_risk_threshold:
                status_color = COLOR_YELLOW
            elif predicted_state == 'active':
                status_color = COLOR_GREEN

            # Display machine details and sensor values
            print(f"{status_color}⚙️  Machine ID: {machine_id:<10} | Predicted State: {predicted_state:<12} | Actual State: {actual_state:<12} | Product: {product_type:<12} | Status: {error_code:<20}{COLOR_RESET}")
            print(f"   Sensors: Temp: {temp: <5.1f}°C | Pressure: {pressure: <5.1f}kPa | Energy: {energy: <5.1f}kWh | Vibr: {vibration: <5.1f}mm/s")
            
            # Display probabilities for each class
            prob_parts = []
            for class_name in ordered_classes:
                # Ensure the column exists before trying to get its value
                if f'prob_{class_name}' in machine:
                    prob = machine[f'prob_{class_name}']
                    prob_parts.append(f"{class_name}: {prob*100:.1f}%")
            print(f"   Probabilities: {' '.join(prob_parts)}")
            print("-" * 80)

    def display_system_stats(self, stats: dict):
        """
        Display aggregate system statistics including runtime and prediction counts.
        
        Args:
            stats (dict): Dictionary containing system statistics
        """
        runtime = datetime.now() - stats['start_time']
        total_seconds = runtime.total_seconds()
        
        logger.info("\n--- SYSTEM STATISTICS ---")
        if total_seconds > 0:
            logger.info(f"   Runtime: {int(total_seconds // 3600)}h {int((total_seconds % 3600) // 60)}m {int(total_seconds % 60)}s")
            logger.info(f"   Total Predictions: {stats['total_predictions']}")
            logger.info(f"   Error Predictions: {stats['error_predictions']}")
            logger.info(f"   High Risk Predictions: {stats['high_risk_predictions']}")
            logger.info(f"   Predictions per Second: {stats['total_predictions']/total_seconds:.2f}")
        else:
            logger.info("   System has just started or no predictions made yet.")
            logger.info(f"   Total Predictions: {stats['total_predictions']}")
        logger.info("-------------------------")