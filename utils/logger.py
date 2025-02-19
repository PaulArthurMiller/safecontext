"""
Logging utility module for SafeContext.

This module provides a centralized logging class to standardize logging across all modules.
Supports multiple output targets (console and file), different log levels, and helper functions.

Usage:
    from utils.logger import SafeContextLogger

    logger = SafeContextLogger(__name__)
    logger.log_event("Processing started")
"""

import logging
import os
from pathlib import Path
from config import config

# Default log directory and file
LOG_DIR = Path(config.cache_dir) / "logs"
LOG_FILE = LOG_DIR / "safecontext.log"

class SafeContextLogger:
    """
    Centralized logger for the SafeContext library.

    Ensures consistent logging behavior across modules with support for
    different log levels and both console & file logging.
    """

    _loggers = {}  # Dictionary to hold logger instances (prevents duplicates)

    logger: logging.Logger

    def __new__(cls, name: str):
        """
        Ensures that the logger instance is reused if already created.

        Args:
            name (str): Name of the logger (usually __name__ from the calling module).

        Returns:
            SafeContextLogger: Configured logger instance.
        """
        if name in cls._loggers:
            return cls._loggers[name]  # ✅ Return the existing SafeContextLogger instance

        # Create a new instance of SafeContextLogger
        instance = super().__new__(cls)
        instance.logger = logging.getLogger(name)
        instance.logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)

        # Define log format
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler (only add if not already present)
        if not any(isinstance(h, logging.StreamHandler) for h in instance.logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            instance.logger.addHandler(console_handler)

        # File handler (only add if not already present)
        if not any(isinstance(h, logging.FileHandler) for h in instance.logger.handlers):
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            file_handler.setFormatter(formatter)
            instance.logger.addHandler(file_handler)

        # Store instance in the class-level dictionary for reuse
        cls._loggers[name] = instance  
        return instance  # Return the SafeContextLogger instance (NOT logging.Logger)

    def log_event(self, message: str):
        """
        Log a general event at INFO level.

        Args:
            message (str): Event description.
        """
        self.logger.info(message)

    def log_warning(self, message: str):
        """
        Log a warning event at WARNING level.

        Args:
            message (str): Warning description.
        """
        self.logger.warning(message)

    def log_error(self, message: str):
        """
        Log an error event at ERROR level.

        Args:
            message (str): Error description.
        """
        self.logger.error(message)

    def log_decision(self, action: str, details: str):
        """
        Log a key decision made during text processing.

        Args:
            action (str): The decision action taken.
            details (str): Additional details about the decision.
        """
        self.logger.info(f"Decision: {action} | Details: {details}")

    # Expose standard logging methods (for convenience)
    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)

if __name__ == "__main__":
    # Example usage
    logger = SafeContextLogger(__name__)

    logger.log_event("Pipeline initialized")
    logger.log_warning("Directive confidence score is high")
    logger.log_error("Failed to classify text chunk")
    logger.log_decision("Sanitized text", "Removed directive elements from chunk 42")
