# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

import sys
from pathlib import Path

from loguru import logger

from coreason_graph_nexus.config import settings

__all__ = ["logger", "configure_logging"]

# Remove default handler
logger.remove()

# Sink 1: Stdout (Human-readable)
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)


def configure_logging() -> None:
    """
    Configures the file logging sink.
    This should be called at the application entry point.
    """
    log_path = Path("logs")
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)  # pragma: no cover

    # Sink 2: File (JSON, Rotation, Retention)
    # We use a hardcoded path relative to CWD as per original design,
    # but now explicit initialization.
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        serialize=True,
        enqueue=True,
        level=settings.log_level,
    )
