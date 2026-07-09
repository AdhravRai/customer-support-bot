"""
Application logging configuration.

This module provides a single, reusable logger instance
for the entire application.

Instead of using print(), every module should import this logger.
"""

import logging
import sys


def setup_logger(name: str = "customer_support_rag") -> logging.Logger:
    """
    Create and configure the application logger.

    Args:
        name: Name of the logger.

    Returns:
        Configured logger instance.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if imported multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


logger = setup_logger()