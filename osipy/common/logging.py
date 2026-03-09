"""Logging configuration for osipy.

This module provides structured logging with configurable verbosity
levels for diagnostics and audit trails.

"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Produces JSON-formatted log entries suitable for log aggregation
    systems and automated parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to format.

        Returns
        -------
        str
            JSON-formatted log entry.
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        for key in ["voxel_index", "processing_time", "convergence_status"]:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Human-readable text log formatter.

    Produces formatted log entries suitable for console output
    and manual inspection.
    """

    def __init__(self) -> None:
        """Initialize text formatter with standard format."""
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def configure_logging(
    level: int = logging.INFO,
    format: str = "text",
    output: str | Path | TextIO | None = None,
) -> logging.Logger:
    """Configure structured logging for osipy.

    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    format : str, default="text"
        Output format: "text" for human-readable, "json" for structured.
    output : str | Path | TextIO | None
        Output destination. None for stdout, path string for file,
        or file-like object for custom output.

    Returns
    -------
    logging.Logger
        Configured root logger for osipy.

    Examples
    --------
    >>> import logging
    >>> from osipy.common.logging import configure_logging
    >>> # Configure text logging to console
    >>> logger = configure_logging(level=logging.INFO, format="text")
    >>> # Configure JSON logging to file
    >>> logger = configure_logging(
    ...     level=logging.DEBUG,
    ...     format="json",
    ...     output="osipy.log",
    ... )

    Notes
    -----
    Verbosity levels:
    - DEBUG: Per-voxel fitting details, intermediate values
    - INFO: Processing stages, progress updates
    - WARNING: Data quality issues, fallback behaviors
    - ERROR: Processing failures, invalid inputs
    """
    # Get or create osipy logger
    logger = logging.getLogger("osipy")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    if format.lower() == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = TextFormatter()

    # Create handler
    handler: logging.Handler
    if output is None:
        handler = logging.StreamHandler(sys.stdout)
    elif isinstance(output, (str, Path)):
        handler = logging.FileHandler(output)
    else:
        handler = logging.StreamHandler(output)

    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Parameters
    ----------
    name : str
        Logger name (typically __name__).

    Returns
    -------
    logging.Logger
        Child logger inheriting osipy configuration.

    Examples
    --------
    >>> from osipy.common.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
    """
    return logging.getLogger(f"osipy.{name}")
