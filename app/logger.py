"""
logger.py - Centralized logging configuration for MSA8770 Legal Demo

Provides structured logging with:
- Console output for development
- File logging with rotation
- Session-based query logs
- Request/response tracking
"""

import logging
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
from contextvars import ContextVar

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Log directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Session logs directory (for individual query sessions)
SESSION_LOG_DIR = LOG_DIR / "sessions"
SESSION_LOG_DIR.mkdir(exist_ok=True)

# Context variable for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="no-request")


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get("no-request"),
        }

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ReadableFormatter(logging.Formatter):
    """Human-readable formatter for console output."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        request_id = request_id_var.get("no-request")[:8]

        # Color codes for different levels
        colors = {
            "DEBUG": "\033[36m",     # Cyan
            "INFO": "\033[32m",      # Green
            "WARNING": "\033[33m",   # Yellow
            "ERROR": "\033[31m",     # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        reset = "\033[0m"
        color = colors.get(record.levelname, "")

        base = f"{timestamp} [{request_id}] {color}{record.levelname:8}{reset} {record.name}: {record.getMessage()}"

        if hasattr(record, "extra_data") and record.extra_data:
            data_str = json.dumps(record.extra_data, default=str)
            if len(data_str) > 200:
                data_str = data_str[:200] + "..."
            base += f" | {data_str}"

        return base


def setup_logging(log_level: str = "INFO") -> None:
    """
    Initialize the logging system.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler (readable format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(ReadableFormatter())
    root_logger.addHandler(console_handler)

    # App log file (JSON format, rotating)
    app_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    app_handler.setLevel(level)
    app_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(app_handler)

    # Error-only log file
    error_handler = RotatingFileHandler(
        LOG_DIR / "error.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info("Logging initialized", extra={"extra_data": {"log_dir": str(LOG_DIR)}})


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def set_request_id(request_id: str) -> None:
    """Set the current request ID in context."""
    request_id_var.set(request_id)


def get_request_id() -> str:
    """Get the current request ID from context."""
    return request_id_var.get("no-request")


class SessionLogger:
    """
    Logger for individual query sessions.

    Saves a complete log of all events for a single query,
    enabling replay and debugging.
    """

    def __init__(self, question: str, session_id: Optional[str] = None):
        self.session_id = session_id or generate_request_id()
        self.question = question
        self.start_time = datetime.utcnow()
        self.events: list = []
        self.metadata: Dict[str, Any] = {}

        # Create session log file
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        safe_question = "".join(c if c.isalnum() or c in " -_" else "_" for c in question[:50])
        self.filename = f"{timestamp}_{self.session_id[:8]}_{safe_question}.json"
        self.filepath = SESSION_LOG_DIR / self.filename

        self._logger = get_logger(f"session.{self.session_id[:8]}")
        self._logger.info(f"Session started: {question[:100]}")

    def log_event(self, event: Dict[str, Any]) -> None:
        """Log an agent event to the session."""
        self.events.append({
            **event,
            "session_time_ms": int((datetime.utcnow() - self.start_time).total_seconds() * 1000)
        })

    def set_metadata(self, key: str, value: Any) -> None:
        """Set session metadata."""
        self.metadata[key] = value

    def save(self) -> str:
        """Save the session log to file and return the filepath."""
        end_time = datetime.utcnow()
        duration_ms = int((end_time - self.start_time).total_seconds() * 1000)

        session_data = {
            "session_id": self.session_id,
            "question": self.question,
            "start_time": self.start_time.isoformat() + "Z",
            "end_time": end_time.isoformat() + "Z",
            "duration_ms": duration_ms,
            "event_count": len(self.events),
            "metadata": self.metadata,
            "events": self.events
        }

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=str)

        self._logger.info(
            f"Session saved: {self.filename}",
            extra={"extra_data": {"duration_ms": duration_ms, "events": len(self.events)}}
        )

        return str(self.filepath)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the session."""
        return {
            "session_id": self.session_id,
            "question": self.question,
            "duration_ms": int((datetime.utcnow() - self.start_time).total_seconds() * 1000),
            "event_count": len(self.events),
            "log_file": self.filename
        }


def log_with_data(logger: logging.Logger, level: int, message: str, data: Dict[str, Any]) -> None:
    """Log a message with additional structured data."""
    logger.log(level, message, extra={"extra_data": data})
