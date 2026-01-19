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
from unittest.mock import MagicMock

import networkx as nx
import pytest
from neo4j.exceptions import ServiceUnavailable
from neo4j.graph import Node, Path, Relationship
from pytest_mock import MockFixture

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient, Neo4jClientAsync


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


def test_verify_connectivity_failure(client: Neo4jClient, mock_driver: Any, mocker: MockFixture) -> None:
    """Test handling of ServiceUnavailable in verify_connectivity."""
    mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Cannot connect")
    mock_logger = mocker.patch("coreason_graph_nexus.adapters.neo4j_adapter.logger")

    with pytest.raises(ServiceUnavailable):
        client.verify_connectivity()

    mock_logger.error.assert_called_once()
    assert "Failed to connect to Neo4j" in mock_logger.error.call_args[0][0]


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


def test_to_networkx_failure(client: Neo4jClient, mock_driver: Any, mocker: MockFixture) -> None:
    """Test handling of exceptions in to_networkx."""
    mock_driver.execute_query.side_effect = Exception("Conversion Failed")
    mock_logger = mocker.patch("coreason_graph_nexus.adapters.neo4j_adapter.logger")

    with pytest.raises(Exception, match="Conversion Failed"):
        client.to_networkx("MATCH ...")

    mock_logger.error.assert_called_once()
    assert "Failed to convert Cypher to NetworkX" in mock_logger.error.call_args[0][0]


def test_to_networkx_complex_types(client: Neo4jClient, mock_driver: Any, mocker: MockFixture) -> None:
    """Test handling of Path objects and lists (nested structures)."""
    # Create nodes
    n1 = mocker.create_autospec(Node, instance=True)
    n1.element_id = "n1"
    n1.labels = {"A"}
    n1.items.return_value = [("p", 1)]

    n2 = mocker.create_autospec(Node, instance=True)
    n2.element_id = "n2"
    n2.labels = {"B"}
    n2.items.return_value = [("p", 2)]

    # Create relationship
    r1 = mocker.create_autospec(Relationship, instance=True)
    r1.start_node.element_id = "n1"
    r1.end_node.element_id = "n2"
    r1.type = "LINK"
    r1.items.return_value = []

    # Mock Path
    path_mock = mocker.create_autospec(Path, instance=True)
    path_mock.nodes = [n1, n2]
    path_mock.relationships = [r1]

    # Mock List of Nodes (e.g. from collect(n))
    list_mock = [n1, n2]

    # Record returning a path and a list
    record_mock = MagicMock()
    record_mock.values.return_value = [path_mock, list_mock]

    mock_driver.execute_query.return_value = ([record_mock], None, None)

    graph = client.to_networkx("MATCH path, collect(n)...")

    assert "n1" in graph.nodes
    assert "n2" in graph.nodes
    assert ("n1", "n2") in graph.edges


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


# --- Async Tests ---


@pytest.fixture
def mock_async_driver(mocker: MockFixture) -> Any:
    driver = mocker.AsyncMock()
    driver.close = mocker.AsyncMock()
    driver.verify_connectivity = mocker.AsyncMock()
    driver.execute_query = mocker.AsyncMock(return_value=([], None, None))
    mocker.patch("neo4j.AsyncGraphDatabase.driver", return_value=driver)
    return driver


@pytest.fixture
async def client_async(mock_async_driver: Any) -> Neo4jClientAsync:
    return Neo4jClientAsync("bolt://localhost:7687", ("user", "pass"))


@pytest.mark.asyncio
async def test_async_initialization(client_async: Neo4jClientAsync, mock_async_driver: Any) -> None:
    assert client_async._uri == "bolt://localhost:7687"
    assert client_async._driver == mock_async_driver


@pytest.mark.asyncio
async def test_async_context_manager(client_async: Neo4jClientAsync, mock_async_driver: Any) -> None:
    async with client_async as c:
        assert c is client_async
        mock_async_driver.verify_connectivity.assert_called_once()
    mock_async_driver.close.assert_called_once()


@pytest.mark.asyncio
async def test_async_execute_query_success(client_async: Neo4jClientAsync, mock_async_driver: Any) -> None:
    mock_record = MagicMock()
    mock_record.data.return_value = {"key": "value"}
    mock_async_driver.execute_query.return_value = ([mock_record], None, None)

    result = await client_async.execute_query("MATCH (n) RETURN n")

    assert result == [{"key": "value"}]
    mock_async_driver.execute_query.assert_called_once()


@pytest.mark.asyncio
async def test_async_batch_write_success(client_async: Neo4jClientAsync, mock_async_driver: Any) -> None:
    data = [{"id": 1}, {"id": 2}, {"id": 3}]
    await client_async.batch_write("UNWIND...", data, batch_size=2)
    assert mock_async_driver.execute_query.call_count == 2


@pytest.mark.asyncio
async def test_async_batch_write_invalid_size(client_async: Neo4jClientAsync) -> None:
    with pytest.raises(ValueError, match="Batch size must be positive"):
        await client_async.batch_write("Q", [], batch_size=0)


@pytest.mark.asyncio
async def test_async_merge_nodes(client_async: Neo4jClientAsync, mock_async_driver: Any) -> None:
    data = [{"id": 1}]
    await client_async.merge_nodes("Label", data, merge_keys=["id"])
    mock_async_driver.execute_query.assert_called_once()
    assert "MERGE" in mock_async_driver.execute_query.call_args[0][0]


@pytest.mark.asyncio
async def test_async_merge_nodes_invalid_keys(client_async: Neo4jClientAsync) -> None:
    with pytest.raises(ValueError, match="merge_keys must not be empty"):
        await client_async.merge_nodes("Label", [{"id": 1}], merge_keys=[])


@pytest.mark.asyncio
async def test_async_merge_relationships(client_async: Neo4jClientAsync, mock_async_driver: Any) -> None:
    data = [{"start": 1, "end": 2}]
    await client_async.merge_relationships("S", "start", "E", "end", "REL", data)
    mock_async_driver.execute_query.assert_called_once()
    assert "MERGE" in mock_async_driver.execute_query.call_args[0][0]


@pytest.mark.asyncio
async def test_async_to_networkx(client_async: Neo4jClientAsync, mock_async_driver: Any, mocker: MockFixture) -> None:
    node_mock = mocker.create_autospec(Node, instance=True)
    node_mock.element_id = "n1"
    node_mock.labels = {"A"}
    node_mock.items.return_value = []

    record_mock = MagicMock()
    record_mock.values.return_value = [node_mock]

    mock_async_driver.execute_query.return_value = ([record_mock], None, None)

    graph = await client_async.to_networkx("MATCH...")
    assert "n1" in graph.nodes


@pytest.mark.asyncio
async def test_async_verify_connectivity_failure(
    client_async: Neo4jClientAsync, mock_async_driver: Any, mocker: MockFixture
) -> None:
    mock_async_driver.verify_connectivity.side_effect = ServiceUnavailable("Fail")
    mock_logger = mocker.patch("coreason_graph_nexus.adapters.neo4j_adapter.logger")

    with pytest.raises(ServiceUnavailable):
        await client_async.verify_connectivity()

    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_async_execute_query_failure(client_async: Neo4jClientAsync, mock_async_driver: Any) -> None:
    mock_async_driver.execute_query.side_effect = Exception("Fail")
    with pytest.raises(Exception, match="Fail"):
        await client_async.execute_query("Q")


@pytest.mark.asyncio
async def test_async_batch_write_failure(
    client_async: Neo4jClientAsync, mock_async_driver: Any, mocker: MockFixture
) -> None:
    mock_async_driver.execute_query.side_effect = Exception("Batch Fail")
    mock_logger = mocker.patch("coreason_graph_nexus.adapters.neo4j_adapter.logger")

    with pytest.raises(Exception, match="Batch Fail"):
        await client_async.batch_write("Q", [{"id": 1}])

    # Called twice: 1 for query failure, 1 for batch failure catch
    assert mock_logger.error.call_count == 2
    mock_logger.error.assert_any_call("Query execution failed: Batch Fail")
    mock_logger.error.assert_any_call("Batch write failed after processing 0 records: Batch Fail")


@pytest.mark.asyncio
async def test_async_to_networkx_failure(
    client_async: Neo4jClientAsync, mock_async_driver: Any, mocker: MockFixture
) -> None:
    mock_async_driver.execute_query.side_effect = Exception("Convert Fail")
    mock_logger = mocker.patch("coreason_graph_nexus.adapters.neo4j_adapter.logger")

    with pytest.raises(Exception, match="Convert Fail"):
        await client_async.to_networkx("Q")

    mock_logger.error.assert_called_once()
