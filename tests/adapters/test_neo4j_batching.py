# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from typing import Any
from unittest.mock import MagicMock, call

import pytest
from neo4j.exceptions import ServiceUnavailable
from pytest_mock import MockerFixture

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient


@pytest.fixture
def mock_driver(mocker: MockerFixture) -> MagicMock:
    """Fixture to mock the Neo4j driver."""
    mock = mocker.patch("neo4j.GraphDatabase.driver")
    # Mock the return value of execute_query
    driver_instance = mock.return_value
    # execute_query returns (records, summary, keys)
    driver_instance.execute_query.return_value = ([], None, None)
    return driver_instance  # type: ignore[no-any-return]


@pytest.fixture
def client(mock_driver: MagicMock) -> Neo4jClient:
    """Fixture to create a Neo4jClient instance."""
    return Neo4jClient("bolt://localhost:7687", ("neo4j", "password"))


def test_batch_write_happy_path_single_batch(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test executing a batch write that fits in a single batch."""
    query = "UNWIND $batch AS row MERGE (n:Test {id: row.id})"
    data: list[dict[str, Any]] = [{"id": 1}, {"id": 2}, {"id": 3}]

    client.batch_write(query, data, batch_size=10)

    mock_driver.execute_query.assert_called_once_with(query, parameters_={"batch": data}, database_="neo4j")


def test_batch_write_happy_path_multiple_batches(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test executing a batch write that requires multiple batches."""
    query = "UNWIND $batch AS row MERGE (n:Test {id: row.id})"
    data: list[dict[str, Any]] = [{"id": i} for i in range(5)]
    batch_size = 2

    client.batch_write(query, data, batch_size=batch_size)

    # Should be called 3 times: [0,1], [2,3], [4]
    assert mock_driver.execute_query.call_count == 3

    expected_calls = [
        call(query, parameters_={"batch": [{"id": 0}, {"id": 1}]}, database_="neo4j"),
        call(query, parameters_={"batch": [{"id": 2}, {"id": 3}]}, database_="neo4j"),
        call(query, parameters_={"batch": [{"id": 4}]}, database_="neo4j"),
    ]
    mock_driver.execute_query.assert_has_calls(expected_calls)


def test_batch_write_empty_data(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test that empty data does not trigger any queries."""
    client.batch_write("QUERY", [], batch_size=100)
    mock_driver.execute_query.assert_not_called()


def test_batch_write_custom_param_name(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test using a custom parameter name."""
    query = "UNWIND $rows AS row MERGE (n:Test {id: row.id})"
    data: list[dict[str, Any]] = [{"id": 1}]

    client.batch_write(query, data, batch_size=10, batch_param_name="rows")

    mock_driver.execute_query.assert_called_once_with(query, parameters_={"rows": data}, database_="neo4j")


def test_batch_write_failure_propagates(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test that an exception in execution raises the error."""
    mock_driver.execute_query.side_effect = ServiceUnavailable("DB Down")

    with pytest.raises(ServiceUnavailable):
        client.batch_write("QUERY", [{"id": 1}])
