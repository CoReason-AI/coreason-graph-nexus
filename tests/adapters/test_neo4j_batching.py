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


def test_batch_write_invalid_batch_size(client: Neo4jClient) -> None:
    """Test that a non-positive batch size raises ValueError."""
    with pytest.raises(ValueError, match="Batch size must be positive"):
        client.batch_write("QUERY", [{"id": 1}], batch_size=0)

    with pytest.raises(ValueError, match="Batch size must be positive"):
        client.batch_write("QUERY", [{"id": 1}], batch_size=-5)


def test_batch_write_exact_multiple(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test when data length is an exact multiple of batch size."""
    query = "QUERY"
    data: list[dict[str, Any]] = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    batch_size = 2

    client.batch_write(query, data, batch_size=batch_size)

    assert mock_driver.execute_query.call_count == 2
    expected_calls = [
        call(query, parameters_={"batch": [{"id": 1}, {"id": 2}]}, database_="neo4j"),
        call(query, parameters_={"batch": [{"id": 3}, {"id": 4}]}, database_="neo4j"),
    ]
    mock_driver.execute_query.assert_has_calls(expected_calls)


def test_batch_write_batch_size_one(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test granular batching with size 1."""
    query = "QUERY"
    data: list[dict[str, Any]] = [{"id": 1}, {"id": 2}]

    client.batch_write(query, data, batch_size=1)

    assert mock_driver.execute_query.call_count == 2


def test_batch_write_complex_nested_data(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test batching with complex nested structures."""
    query = "UNWIND $batch AS row MERGE (n:Complex {prop: row.props})"
    complex_item = {"id": "A1", "metadata": {"source": "test", "tags": ["a", "b"]}, "metrics": [1.0, 2.5, 3.1]}
    data = [complex_item]

    client.batch_write(query, data, batch_size=1)

    mock_driver.execute_query.assert_called_once_with(query, parameters_={"batch": [complex_item]}, database_="neo4j")


def test_batch_write_partial_failure(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test that execution stops if a middle batch fails."""
    query = "QUERY"
    data: list[dict[str, Any]] = [{"id": 1}, {"id": 2}, {"id": 3}]

    # First call succeeds, second fails
    mock_driver.execute_query.side_effect = [([], None, None), ServiceUnavailable("Network Error"), ([], None, None)]

    with pytest.raises(ServiceUnavailable, match="Network Error"):
        client.batch_write(query, data, batch_size=1)

    # Should have attempted 2 calls, not 3
    assert mock_driver.execute_query.call_count == 2


# --- Tests for merge_nodes ---


def test_merge_nodes_query_construction(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test that merge_nodes constructs the correct Cypher query."""
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

    client.merge_nodes(label="Person", data=data, merge_keys=["id"], batch_size=100)

    expected_query = "UNWIND $batch AS row MERGE (n:`Person` { `id`: row.`id` }) SET n += row"

    mock_driver.execute_query.assert_called_once_with(expected_query, parameters_={"batch": data}, database_="neo4j")


def test_merge_nodes_multiple_keys(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test merge_nodes with multiple composite keys."""
    data = [{"firstName": "Alice", "lastName": "Smith"}]

    client.merge_nodes(label="Person", data=data, merge_keys=["firstName", "lastName"])

    expected_query = (
        "UNWIND $batch AS row "
        "MERGE (n:`Person` { `firstName`: row.`firstName`, `lastName`: row.`lastName` }) "
        "SET n += row"
    )

    mock_driver.execute_query.assert_called_once_with(expected_query, parameters_={"batch": data}, database_="neo4j")


def test_merge_nodes_no_keys_raises(client: Neo4jClient) -> None:
    """Test that calling merge_nodes without keys raises ValueError."""
    with pytest.raises(ValueError, match="merge_keys must not be empty"):
        client.merge_nodes("Person", [{"id": 1}], [])


# --- Tests for merge_relationships ---


def test_merge_relationships_query_construction(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test that merge_relationships constructs the correct Cypher query with separated keys and properties."""
    data = [{"start_ref": 1, "end_ref": 2, "weight": 0.5}]

    client.merge_relationships(
        start_label="Person",
        start_data_key="start_ref",
        end_label="Company",
        end_data_key="end_ref",
        rel_type="WORKS_FOR",
        data=data,
        start_node_prop="id",  # explicitly using default just to be clear
        end_node_prop="id",
        batch_size=100,
    )

    expected_query = (
        "UNWIND $batch AS row "
        "MATCH (source:`Person` { `id`: row.`start_ref` }) "
        "MATCH (target:`Company` { `id`: row.`end_ref` }) "
        "MERGE (source)-[r:`WORKS_FOR`]->(target) "
        "SET r += row"
    )

    mock_driver.execute_query.assert_called_once_with(expected_query, parameters_={"batch": data}, database_="neo4j")


def test_merge_relationships_custom_props(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test merge_relationships with custom node properties (not 'id')."""
    data = [{"u_uuid": "abc", "c_code": "xyz"}]

    client.merge_relationships(
        start_label="User",
        start_data_key="u_uuid",
        end_label="Country",
        end_data_key="c_code",
        rel_type="LIVES_IN",
        data=data,
        start_node_prop="uuid",
        end_node_prop="iso_code",
    )

    expected_query = (
        "UNWIND $batch AS row "
        "MATCH (source:`User` { `uuid`: row.`u_uuid` }) "
        "MATCH (target:`Country` { `iso_code`: row.`c_code` }) "
        "MERGE (source)-[r:`LIVES_IN`]->(target) "
        "SET r += row"
    )

    mock_driver.execute_query.assert_called_once_with(expected_query, parameters_={"batch": data}, database_="neo4j")


def test_merge_relationships_empty_data(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test that merge_relationships handles empty data gracefully."""
    client.merge_relationships("A", "k", "B", "k", "REL", [], batch_size=100)
    mock_driver.execute_query.assert_not_called()


# --- Edge Case & Complex Scenario Tests ---


def test_merge_nodes_escaping(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test that labels and keys with spaces/special chars are escaped with backticks."""
    data = [{"User Group": "Admins", "Full Name": "Admin User"}]

    client.merge_nodes(label="User Group", data=data, merge_keys=["Full Name"], batch_size=10)

    expected_query = "UNWIND $batch AS row MERGE (n:`User Group` { `Full Name`: row.`Full Name` }) SET n += row"

    mock_driver.execute_query.assert_called_once_with(expected_query, parameters_={"batch": data}, database_="neo4j")


def test_merge_relationships_escaping(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test that relationship types and labels with special chars are escaped."""
    data = [{"s": 1, "e": 2}]

    client.merge_relationships(
        start_label="Start Node",
        start_data_key="key-1",
        end_label="End Node",
        end_data_key="key-2",
        rel_type="HAS CONNECTION",
        data=data,
        start_node_prop="id",
        end_node_prop="id",
    )

    expected_query = (
        "UNWIND $batch AS row "
        "MATCH (source:`Start Node` { `id`: row.`key-1` }) "
        "MATCH (target:`End Node` { `id`: row.`key-2` }) "
        "MERGE (source)-[r:`HAS CONNECTION`]->(target) "
        "SET r += row"
    )

    mock_driver.execute_query.assert_called_once_with(expected_query, parameters_={"batch": data}, database_="neo4j")


def test_merge_nodes_complex_properties(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test that complex property types (lists, nested dicts) are passed correctly."""
    complex_data = [
        {"id": 1, "tags": ["a", "b", "c"], "metadata": {"created": "today", "author": "me"}, "scores": [0.1, 0.9]}
    ]

    client.merge_nodes(label="Item", data=complex_data, merge_keys=["id"])

    # Verify the query structure is standard
    expected_query = "UNWIND $batch AS row MERGE (n:`Item` { `id`: row.`id` }) SET n += row"

    # Verify the complex data was passed in the parameters
    mock_driver.execute_query.assert_called_once_with(
        expected_query, parameters_={"batch": complex_data}, database_="neo4j"
    )


def test_merge_nodes_empty_data_does_not_error(client: Neo4jClient, mock_driver: MagicMock) -> None:
    """Test that calling merge_nodes with empty data list does not raise error."""
    client.merge_nodes(label="Person", data=[], merge_keys=["id"])
    mock_driver.execute_query.assert_not_called()


def test_merge_nodes_empty_data_invalid_keys(client: Neo4jClient) -> None:
    """Test that invalid keys raise error even if data is empty (validation first)."""
    with pytest.raises(ValueError, match="merge_keys must not be empty"):
        client.merge_nodes("Person", [], [])
