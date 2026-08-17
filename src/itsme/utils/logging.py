"""
Structured Logging Infrastructure for ItsMe Pipeline.
"""

import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger instance for a given module/stage name.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger

def log_stage_event(
    logger: logging.Logger,
    stage: str,
    status: str,
    file_id: str | None = None,
    duration: float | None = None,
    error: str | None = None,
    level: str = "INFO"
):
    """
    Log structured pipeline stage event.
    Example: 2026-08-18 12:00:01 INFO dataset.audio processed utt_000001
    """
    msg_parts = [stage, status]
    if file_id:
        msg_parts.append(file_id)
    if duration is not None:
        msg_parts.append(f"duration={duration:.2f}s")
    if error:
        msg_parts.append(f"error='{error}'")
        
    message = " ".join(msg_parts)
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)
