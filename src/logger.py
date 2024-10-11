import logging
import os
from datetime import datetime

# Create a unique log file based on the current date (year-month-day)
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Define the directory where logs will be stored
LOG_DIR = "logs"

# Ensure the log directory exists; create it if not
os.makedirs(LOG_DIR, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Set up basic logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,         # Log to the file path
    filemode="w",                   # Overwrite the log file for each new run
    format="%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO              # Set default logging level to INFO
)

# Example logging
logging.info("This is an info message.")
#logging.warning("This is a warning message.")
#logging.error("This is an error message.")

if __name__=="__main__":
    logging.info("logging has started")
