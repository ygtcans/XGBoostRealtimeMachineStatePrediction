# realtime_machine_state_predict/main.py

import asyncio
from datetime import datetime

# Import the CONFIG and the main system class
from config import CONFIG, logger
from system_orchestrator import RealTimePredictionSystem

async def main():
    """Main application entry point."""
    print("🚀 REAL-TIME MACHINE STATE PREDICTION SYSTEM")
    print("=" * 60)
    # Use f-strings for direct embedding of CONFIG values
    print(f"📡 WebSocket URL: {CONFIG['WS_URL']}")
    print(f"🎯 Error Threshold: {CONFIG['OPTIMAL_ERROR_THRESHOLD']:.4f}") # Format for clarity
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\nInitializing system...")
    
    system = RealTimePredictionSystem(CONFIG)
    if system.running:
        await system.run()
    else:
        logger.critical("System failed to initialize. Exiting.")

if __name__ == "__main__":
    asyncio.run(main())