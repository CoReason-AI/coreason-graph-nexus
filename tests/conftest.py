# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_driver() -> Generator[MagicMock, None, None]:
    """Mock the neo4j.GraphDatabase.driver."""
    with patch("coreason_graph_nexus.adapters.neo4j_adapter.GraphDatabase.driver") as mock:
        yield mock


@pytest.fixture
def mock_async_driver() -> Generator[MagicMock, None, None]:
    """Mock the neo4j.AsyncGraphDatabase.driver."""
    with patch("coreason_graph_nexus.adapters.neo4j_adapter.AsyncGraphDatabase.driver") as mock:
        yield mock
