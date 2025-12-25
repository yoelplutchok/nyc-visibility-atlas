"""
JSONL structured logging utilities.

This module provides structured logging that writes to JSONL files,
one per script run, as required by the pipeline contract.

Each log entry includes:
- timestamp
- run_id (unique per script execution)
- level (INFO, WARNING, ERROR, etc.)
- event type
- message
- additional context
"""

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from visibility_atlas.paths import paths, ensure_dir


# =============================================================================
# Run ID generation
# =============================================================================

def generate_run_id() -> str:
    """
    Generate a unique run ID for this script execution.
    
    Format: YYYYMMDD_HHMMSS_<short_uuid>
    
    Returns:
        Unique run identifier string.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


# =============================================================================
# JSONL Handler
# =============================================================================

class JSONLHandler(logging.Handler):
    """
    A logging handler that writes structured JSON lines to a file.
    """
    
    def __init__(self, log_path: Path, run_id: str):
        super().__init__()
        self.log_path = log_path
        self.run_id = run_id
        self._file = None
    
    def _ensure_file(self):
        """Lazily open the log file."""
        if self._file is None:
            ensure_dir(self.log_path.parent)
            self._file = open(self.log_path, "a", encoding="utf-8")
    
    def emit(self, record: logging.LogRecord):
        """Write a log record as a JSON line."""
        try:
            self._ensure_file()
            
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            
            # Add extra fields if present
            if hasattr(record, "event_type"):
                log_entry["event_type"] = record.event_type
            if hasattr(record, "context"):
                log_entry["context"] = record.context
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.format(record)
            
            self._file.write(json.dumps(log_entry) + "\n")
            self._file.flush()
            
        except Exception:
            self.handleError(record)
    
    def close(self):
        """Close the log file."""
        if self._file is not None:
            self._file.close()
            self._file = None
        super().close()


# =============================================================================
# Logger setup
# =============================================================================

_LOGGERS: dict[str, logging.Logger] = {}
_RUN_ID: str | None = None


def get_run_id() -> str:
    """Get the current run ID, generating one if needed."""
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = generate_run_id()
    return _RUN_ID


def set_run_id(run_id: str) -> None:
    """Set a specific run ID (useful for testing or continuation)."""
    global _RUN_ID
    _RUN_ID = run_id


def get_logger(
    script_name: str,
    run_id: str | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Get or create a logger for a pipeline script.
    
    Creates both console and JSONL file handlers.
    
    Args:
        script_name: Name of the script (e.g., "00_build_geographies")
        run_id: Optional run ID; if None, generates or reuses existing.
        console_level: Logging level for console output.
        file_level: Logging level for JSONL file output.
        
    Returns:
        Configured Logger instance.
    """
    if run_id is None:
        run_id = get_run_id()
    else:
        set_run_id(run_id)
    
    logger_key = f"{script_name}_{run_id}"
    
    if logger_key in _LOGGERS:
        return _LOGGERS[logger_key]
    
    # Create logger
    logger = logging.getLogger(logger_key)
    logger.setLevel(min(console_level, file_level))
    logger.handlers = []  # Clear any existing handlers
    
    # Console handler with simple formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # JSONL file handler
    log_file = paths.logs / f"{script_name}_{run_id}.jsonl"
    jsonl_handler = JSONLHandler(log_file, run_id)
    jsonl_handler.setLevel(file_level)
    logger.addHandler(jsonl_handler)
    
    _LOGGERS[logger_key] = logger
    
    # Log initialization
    logger.info(f"Logger initialized for {script_name}", extra={
        "event_type": "logger_init",
        "context": {"script_name": script_name, "run_id": run_id}
    })
    
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    event_type: str,
    **context: Any
) -> None:
    """
    Log a structured event with type and context.
    
    Args:
        logger: Logger instance.
        level: Logging level (e.g., logging.INFO).
        message: Human-readable message.
        event_type: Event type for structured parsing.
        **context: Additional context key-value pairs.
    """
    logger.log(level, message, extra={
        "event_type": event_type,
        "context": context
    })


def log_step_start(logger: logging.Logger, step_name: str, **context: Any) -> None:
    """Log the start of a processing step."""
    log_event(logger, logging.INFO, f"Starting: {step_name}", "step_start", 
              step_name=step_name, **context)


def log_step_end(logger: logging.Logger, step_name: str, **context: Any) -> None:
    """Log the end of a processing step."""
    log_event(logger, logging.INFO, f"Completed: {step_name}", "step_end",
              step_name=step_name, **context)


def log_qa_check(
    logger: logging.Logger,
    check_name: str,
    passed: bool,
    details: str | None = None,
    **context: Any
) -> None:
    """
    Log a QA check result.
    
    Args:
        logger: Logger instance.
        check_name: Name of the QA check.
        passed: Whether the check passed.
        details: Optional details about the check.
        **context: Additional context.
    """
    status = "PASSED" if passed else "FAILED"
    level = logging.INFO if passed else logging.ERROR
    message = f"QA Check [{check_name}]: {status}"
    if details:
        message += f" - {details}"
    
    log_event(logger, level, message, "qa_check",
              check_name=check_name, passed=passed, details=details, **context)


def log_output_written(
    logger: logging.Logger,
    output_path: str | Path,
    row_count: int | None = None,
    **context: Any
) -> None:
    """Log that an output file was written."""
    message = f"Output written: {output_path}"
    if row_count is not None:
        message += f" ({row_count:,} rows)"
    
    log_event(logger, logging.INFO, message, "output_written",
              output_path=str(output_path), row_count=row_count, **context)

