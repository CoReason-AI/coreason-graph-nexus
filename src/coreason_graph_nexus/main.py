# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from coreason_graph_nexus.utils.logger import configure_logging, logger


def hello_world() -> str:
    # Initialize logging (idempotent setup for file sink)
    configure_logging()

    logger.info("Hello World!")
    return "Hello World!"
