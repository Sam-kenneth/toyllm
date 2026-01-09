import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_file_path='logs/pipeline_run.log', level=logging.INFO, max_bytes=10485760, backup_count=5):
    """
    Configure logging with file and console handlers.
    
    Args:
        log_file_path: Path to the log file
        level: Logging level (default: INFO)
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
    
    Returns:
        Configured root logger
    """
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(level)

    # Use RotatingFileHandler to prevent log files from growing too large
    file_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=max_bytes, 
        backupCount=backup_count
    )
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.info(f"Logging system configured. Output saved to '{log_file_path}'.")
    return root_logger