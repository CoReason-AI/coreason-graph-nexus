# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from coreason_graph_nexus.utils.logger import configure_logging


def test_configure_logging_creates_directory() -> None:
    """Test that configure_logging creates the logs directory."""
    test_log_dir = Path("logs_test_dir")

    # Clean up before test
    if test_log_dir.exists():
        shutil.rmtree(test_log_dir)

    with patch("coreason_graph_nexus.utils.logger.Path") as mock_path:
        # We mock Path to avoid actual FS creation in standard locations,
        # but we want to verify the logic.
        # However, to test side-effects correctly, we might use a temporary dir.

        # Real FS test in temporary location
        mock_path_obj = MagicMock(wraps=Path)
        mock_path.return_value = mock_path_obj

        # We intercept "logs" and redirect to a test path?
        # The code hardcodes Path("logs").
        # So we must mock Path to return our test path when initialized with "logs"
        pass

    # Simpler approach: Mock Path and verify mkdir called
    with patch("coreason_graph_nexus.utils.logger.Path") as MockPath:
        mock_instance = MagicMock()
        MockPath.return_value = mock_instance
        mock_instance.exists.return_value = False

        configure_logging()

        MockPath.assert_called_with("logs")
        mock_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
