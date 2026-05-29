"""Structured JSON logging with rotation and correlation ID support.

Provides structured JSON logging using Python's standard logging module with:
- Custom JSON formatter outputting timestamp, level, logger_name, message, correlation_id, extras
- RotatingFileHandler with 100MB max size and 10 backup files
- Console handler with optional colored output for development
- Correlation ID propagation via contextvars for async request tracing

Requirements: 18.3
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import orjson

# Correlation ID context variable - propagates across async calls
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Constants
LOG_FILE_MAX_BYTES = 104_857_600  # 100MB
LOG_FILE_BACKUP_COUNT = 10
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "trading_system.log"


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str | None) -> None:
    """Set the correlation ID in the current context."""
    correlation_id_var.set(correlation_id)


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON.

    Each log line contains:
    - timestamp: ISO 8601 format in UTC
    - level: log level name
    - logger_name: name of the logger
    - message: formatted log message
    - correlation_id: request tracing ID from contextvars (null if not set)
    - Any extra fields passed via the `extra` dict
    """

    # Fields that are part of the standard LogRecord and should not be included as extras
    _RESERVED_ATTRS = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_entry["stack_info"] = record.stack_info

        # Add any extra fields that aren't standard LogRecord attributes
        for key, value in record.__dict__.items():
            if key not in self._RESERVED_ATTRS and not key.startswith("_"):
                log_entry[key] = value

        return orjson.dumps(log_entry, default=str).decode("utf-8")


class ColoredConsoleFormatter(logging.Formatter):
    """Formatter with ANSI color codes for console output in development.

    Colors:
    - DEBUG: cyan
    - INFO: green
    - WARNING: yellow
    - ERROR: red
    - CRITICAL: bold red
    """

    COLORS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format with color codes based on log level."""
        color = self.COLORS.get(record.levelno, "")
        correlation_id = get_correlation_id()
        cid_str = f" [{correlation_id}]" if correlation_id else ""

        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]

        formatted = (
            f"{color}{timestamp} | {record.levelname:<8}{self.RESET}"
            f"{cid_str} | {record.name} | {record.getMessage()}"
        )

        if record.exc_info and record.exc_info[0] is not None:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logging(
    level: str = "INFO",
    log_dir: str = DEFAULT_LOG_DIR,
    log_file: str = DEFAULT_LOG_FILE,
    enable_console: bool = True,
    enable_file: bool = True,
    use_colors: bool = True,
) -> None:
    """Configure the root logger with JSON file handler and optional console handler.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. Created if it doesn't exist.
        log_file: Name of the log file.
        enable_console: Whether to add a console (stderr) handler.
        enable_file: Whether to add a rotating file handler.
        use_colors: Whether to use colored output on console (ignored if enable_console=False).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Get the root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # File handler with JSON formatting and rotation
    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_path / log_file),
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Console handler for development
    if enable_console:
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setLevel(log_level)

        if use_colors and sys.stderr.isatty():
            console_handler.setFormatter(ColoredConsoleFormatter())
        else:
            # Use JSON format for non-TTY console output (e.g., Docker, CI)
            console_handler.setFormatter(JSONFormatter())

        root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with the given name.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A logging.Logger instance configured by setup_logging().
    """
    return logging.getLogger(name)
