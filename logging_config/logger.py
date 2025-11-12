"""
Centralized logging configuration.

Usage:
    from logging_config.logger import get_logger
    logger = get_logger(__name__)
    logger.info("This is info")
    logger.debug("This only shows when VERBOSE=true")
"""
import logging
import sys
from pathlib import Path
from config.settings import LOG_LEVEL, VERBOSE


def get_logger(name: str) -> logging.Logger:
    """
    Get configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Usage:
        logger = get_logger(__name__)
        logger.info("Info message")        # Always shows if LOG_LEVEL <= INFO
        logger.debug("Debug message")      # Only shows if VERBOSE=true
        logger.warning("Warning message")  # Always shows
        logger.error("Error message")      # Always shows
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Determine effective log level
        if VERBOSE:
            effective_level = logging.DEBUG
        else:
            effective_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

        logger.setLevel(effective_level)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(effective_level)

        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # Add handler
        logger.addHandler(console_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


# Module-level logger for this package
logger = get_logger(__name__)
