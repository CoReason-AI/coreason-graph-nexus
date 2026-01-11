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
from neo4j.exceptions import ServiceUnavailable

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient


@pytest.fixture
def mock_driver() -> Generator[MagicMock, None, None]:
    """Mock the neo4j.GraphDatabase.driver."""
    with patch("coreason_graph_nexus.adapters.neo4j_adapter.GraphDatabase.driver") as mock:
        yield mock


def test_initialization(mock_driver: MagicMock) -> None:
    """Test that the client initializes the driver correctly."""
    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    mock_driver.assert_called_once_with("bolt://localhost:7687", auth=("user", "pass"))
    assert client._database == "neo4j"


def test_context_manager(mock_driver: MagicMock) -> None:
    """Test that the context manager calls verify_connectivity and close."""
    driver_instance = mock_driver.return_value

    with Neo4jClient("bolt://localhost:7687", ("user", "pass")) as client:
        driver_instance.verify_connectivity.assert_called_once()
        assert isinstance(client, Neo4jClient)

    driver_instance.close.assert_called_once()


def test_verify_connectivity_failure(mock_driver: MagicMock) -> None:
    """Test that verify_connectivity raises ServiceUnavailable on failure."""
    driver_instance = mock_driver.return_value
    driver_instance.verify_connectivity.side_effect = ServiceUnavailable("Connection failed")

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    with pytest.raises(ServiceUnavailable):
        client.verify_connectivity()


def test_execute_query_success(mock_driver: MagicMock) -> None:
    """Test successful query execution."""
    driver_instance = mock_driver.return_value

    # Mock the return value of execute_query
    # It returns (records, summary, keys)
    mock_record = MagicMock()
    mock_record.data.return_value = {"key": "value"}
    driver_instance.execute_query.return_value = ([mock_record], None, None)

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
    result = client.execute_query("MATCH (n) RETURN n")

    driver_instance.execute_query.assert_called_once_with(
        "MATCH (n) RETURN n",
        parameters_={},
        database_="neo4j",
    )
    assert result == [{"key": "value"}]


def test_execute_query_with_params(mock_driver: MagicMock) -> None:
    """Test query execution with parameters."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.return_value = ([], None, None)

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
    client.execute_query("CREATE (n {name: $name})", {"name": "Test"})

    driver_instance.execute_query.assert_called_once_with(
        "CREATE (n {name: $name})",
        parameters_={"name": "Test"},
        database_="neo4j",
    )


def test_execute_query_failure(mock_driver: MagicMock) -> None:
    """Test query execution handles exceptions."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.side_effect = Exception("Query Error")

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    with pytest.raises(Exception, match="Query Error"):
        client.execute_query("BAD QUERY")
