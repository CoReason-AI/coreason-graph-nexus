# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

import os
from unittest.mock import patch

from coreason_graph_nexus.config import Settings


def test_settings_defaults() -> None:
    """Test default values of Settings."""
    settings = Settings()
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.log_level == "INFO"
    assert settings.default_batch_size == 10000


def test_settings_from_env() -> None:
    """Test that environment variables override defaults."""
    with patch.dict(os.environ, {"NEO4J_URI": "bolt://test:1234", "DEFAULT_BATCH_SIZE": "500"}):
        settings = Settings()
        assert settings.neo4j_uri == "bolt://test:1234"
        assert settings.default_batch_size == 500
