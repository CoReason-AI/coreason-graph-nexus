# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

import pytest
import networkx as nx
from unittest.mock import MagicMock
from pytest_mock import MockFixture
from typing import Any

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient
from neo4j.graph import Node, Relationship


@pytest.fixture
def mock_driver(mocker: MockFixture) -> Any:
    driver = mocker.Mock()
    # Mock close method
    driver.close = mocker.Mock()

    # Mock verify_connectivity
    driver.verify_connectivity = mocker.Mock()

    # Mock execute_query
    driver.execute_query = mocker.Mock(return_value=([], None, None))

    # Patch GraphDatabase.driver to return our mock
    mocker.patch("neo4j.GraphDatabase.driver", return_value=driver)

    return driver


@pytest.fixture
def client(mock_driver: Any) -> Neo4jClient:
    return Neo4jClient("bolt://localhost:7687", ("user", "pass"))


def test_initialization(client: Neo4jClient, mock_driver: Any) -> None:
    assert client._uri == "bolt://localhost:7687"
    assert client._driver == mock_driver


def test_context_manager(client: Neo4jClient, mock_driver: Any) -> None:
    with client as c:
        assert c is client
        mock_driver.verify_connectivity.assert_called_once()

    mock_driver.close.assert_called_once()


def test_execute_query_success(client: Neo4jClient, mock_driver: Any) -> None:
    mock_record = MagicMock()
    mock_record.data.return_value = {"key": "value"}
    mock_driver.execute_query.return_value = ([mock_record], None, None)

    result = client.execute_query("MATCH (n) RETURN n")

    assert result == [{"key": "value"}]
    mock_driver.execute_query.assert_called_once()


def test_execute_query_failure(client: Neo4jClient, mock_driver: Any) -> None:
    mock_driver.execute_query.side_effect = Exception("Query Failed")

    with pytest.raises(Exception, match="Query Failed"):
        client.execute_query("BAD QUERY")


def test_batch_write_success(client: Neo4jClient, mock_driver: Any) -> None:
    data = [{"id": 1}, {"id": 2}, {"id": 3}]

    client.batch_write("UNWIND $batch AS row ...", data, batch_size=2)

    # Should be called twice (batch of 2, batch of 1)
    assert mock_driver.execute_query.call_count == 2


def test_batch_write_empty(client: Neo4jClient, mock_driver: Any) -> None:
    client.batch_write("...", [])
    mock_driver.execute_query.assert_not_called()


def test_merge_nodes(client: Neo4jClient, mock_driver: Any) -> None:
    data = [{"id": 1, "name": "A"}]
    client.merge_nodes("Person", data, merge_keys=["id"])

    mock_driver.execute_query.assert_called_once()
    args, kwargs = mock_driver.execute_query.call_args
    query = args[0]
    assert "MERGE (n:`Person` { `id`: row.`id` })" in query


def test_merge_relationships(client: Neo4jClient, mock_driver: Any) -> None:
    data = [{"start": 1, "end": 2}]
    client.merge_relationships("Person", "start", "Person", "end", "KNOWS", data)

    mock_driver.execute_query.assert_called_once()
    args, kwargs = mock_driver.execute_query.call_args
    query = args[0]
    assert "MATCH (source:`Person`" in query
    assert "MATCH (target:`Person`" in query
    assert "MERGE (source)-[r:`KNOWS`]->(target)" in query


def test_to_networkx_basic(client: Neo4jClient, mock_driver: Any, mocker: MockFixture) -> None:
    # Setup mock records
    node_mock = mocker.create_autospec(Node, instance=True)
    node_mock.element_id = "n1"
    node_mock.labels = {"Person"}
    node_mock.items.return_value = [("name", "Alice")]

    rel_mock = mocker.create_autospec(Relationship, instance=True)
    rel_mock.start_node.element_id = "n1"
    rel_mock.end_node.element_id = "n2"
    rel_mock.type = "KNOWS"
    rel_mock.items.return_value = [("since", 2022)]

    # We need a record that behaves like a mapping
    record_mock = MagicMock()
    record_mock.values.return_value = [node_mock, rel_mock]

    mock_driver.execute_query.return_value = ([record_mock], None, None)

    graph = client.to_networkx("MATCH ...")

    assert isinstance(graph, nx.DiGraph)
    assert "n1" in graph.nodes
    assert graph.nodes["n1"]["name"] == "Alice"
    assert ("n1", "n2") in graph.edges
    assert graph.edges["n1", "n2"]["since"] == 2022


def test_to_networkx_legacy_id_fallback(client: Neo4jClient, mock_driver: Any, mocker: MockFixture) -> None:
    """Test fallback to .id when .element_id is missing (Neo4j 4.x compat)."""
    # Patch the Node class in the adapter so `isinstance(x, Node)` passes for our mock
    mocker.patch("coreason_graph_nexus.adapters.neo4j_adapter.Node", new=MagicMock)

    node_mock = MagicMock()
    del node_mock.element_id  # Ensure it doesn't have it
    node_mock.id = 123
    node_mock.labels = ["Legacy"]
    node_mock.items.return_value = [("prop", "val")]

    record_mock = MagicMock()
    record_mock.values.return_value = [node_mock]

    mock_driver.execute_query.return_value = ([record_mock], None, None)

    graph = client.to_networkx("MATCH ...")

    assert 123 in graph.nodes
    assert graph.nodes[123]["prop"] == "val"
