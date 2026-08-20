"""
Application logging to data folder.

All logs from application start through OCR completion are written to
data/logs/ so they are stored in one place. Call setup_logging() at
application start (e.g. from main_trt_demo main() or TrailerVisionApp.__init__).
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Marker attribute to avoid adding duplicate handlers
_APP_LOG_HANDLER_ATTR = "_edge_orion_data_log_handler"
_LOGS_SETUP = False


def _data_logs_dir(base_dir: str = "data") -> Path:
    """Return data/logs path and ensure it exists."""
    root = Path(base_dir).resolve()
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def setup_logging(
    base_dir: str = "data",
    log_level: int = logging.INFO,
    to_console: bool = True,
    daily_file: bool = True,
) -> None:
    """
    Configure application logging to write to data/logs/.

    Idempotent: safe to call multiple times; only configures once.

    Args:
        base_dir: Base directory (default "data"); logs go to base_dir/logs/.
        log_level: Logging level (default INFO).
        to_console: If True, also emit to stderr (default True).
        daily_file: If True, use a daily log file app_YYYYMMDD.log; else app.log.
    """
    global _LOGS_SETUP
    if _LOGS_SETUP:
        return

    logs_dir = _data_logs_dir(base_dir)
    if daily_file:
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = logs_dir / f"app_{date_str}.log"
    else:
        log_file = logs_dir / "app.log"

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    setattr(file_handler, _APP_LOG_HANDLER_ATTR, True)

    root = logging.getLogger()
    # Avoid duplicate file handlers (e.g. when app is created from different entry points)
    for h in getattr(root, "handlers", []):
        if getattr(h, _APP_LOG_HANDLER_ATTR, False):
            _LOGS_SETUP = True
            return
    root.addHandler(file_handler)

    if to_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    root.setLevel(log_level)
    _LOGS_SETUP = True

    # Log that we're now writing to data folder
    logger = logging.getLogger("app")
    logger.info("Logging to data folder: %s", log_file)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module/component (e.g. __name__)."""
    return logging.getLogger(name if name.startswith("app") else f"app.{name}")


def ensure_logging_setup() -> None:
    """Ensure setup_logging has been called (e.g. from __init__)."""
    setup_logging()
