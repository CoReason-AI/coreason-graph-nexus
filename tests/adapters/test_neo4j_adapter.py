# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from typing import cast
from unittest.mock import MagicMock

import networkx as nx
import pytest
from neo4j.exceptions import ServiceUnavailable
from neo4j.graph import Node, Path, Relationship
from pytest_mock import MockerFixture

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient


@pytest.fixture
def mock_driver(mocker: MockerFixture) -> MagicMock:
    driver = mocker.AsyncMock()
    driver.verify_connectivity = mocker.AsyncMock()
    driver.close = mocker.AsyncMock()
    driver.execute_query = mocker.AsyncMock()
    mocker.patch("neo4j.AsyncGraphDatabase.driver", return_value=driver)
    return cast(MagicMock, driver)


@pytest.mark.asyncio
async def test_lifecycle(mock_driver: MagicMock) -> None:
    client = Neo4jClient("bolt://host", ("u", "p"))
    async with client:
        mock_driver.verify_connectivity.assert_awaited_once()
    mock_driver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_connectivity_error(mock_driver: MagicMock) -> None:
    mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Down")
    client = Neo4jClient("bolt://host", ("u", "p"))
    with pytest.raises(ServiceUnavailable):
        async with client:
            pass


@pytest.mark.asyncio
async def test_execute_query_success(mock_driver: MagicMock) -> None:
    # Mock records
    record = MagicMock()
    record.data.return_value = {"key": "value"}
    mock_driver.execute_query.return_value = ([record], None, None)

    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        result = await client.execute_query("MATCH (n) RETURN n")
        assert result == [{"key": "value"}]


@pytest.mark.asyncio
async def test_execute_query_failure(mock_driver: MagicMock) -> None:
    mock_driver.execute_query.side_effect = Exception("Cypher Error")
    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        with pytest.raises(Exception, match="Cypher Error"):
            await client.execute_query("BAD QUERY")


@pytest.mark.asyncio
async def test_batch_write(mock_driver: MagicMock) -> None:
    mock_driver.execute_query.return_value = ([], None, None)
    data = [{"id": i} for i in range(15)]

    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        await client.batch_write("UNWIND $batch ...", data, batch_size=10)

    assert mock_driver.execute_query.await_count == 2
    # Check batch sizes
    calls = mock_driver.execute_query.await_args_list
    assert len(calls[0].kwargs["parameters_"]["batch"]) == 10
    assert len(calls[1].kwargs["parameters_"]["batch"]) == 5


@pytest.mark.asyncio
async def test_batch_write_error(mock_driver: MagicMock) -> None:
    mock_driver.execute_query.side_effect = Exception("Write Error")
    data = [{"id": 1}]
    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        with pytest.raises(Exception, match="Write Error"):
            await client.batch_write("Q", data)


@pytest.mark.asyncio
async def test_batch_write_invalid_size(mock_driver: MagicMock) -> None:
    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        with pytest.raises(ValueError):
            await client.batch_write("Q", [], batch_size=0)


@pytest.mark.asyncio
async def test_merge_nodes(mock_driver: MagicMock) -> None:
    mock_driver.execute_query.return_value = ([], None, None)
    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        await client.merge_nodes("Person", [{"id": 1}], merge_keys=["id"])

    mock_driver.execute_query.assert_awaited_once()
    query = mock_driver.execute_query.await_args[0][0]
    assert "MERGE (n:`Person` { `id`: row.`id` })" in query


@pytest.mark.asyncio
async def test_merge_nodes_empty_keys(mock_driver: MagicMock) -> None:
    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        with pytest.raises(ValueError):
            await client.merge_nodes("Person", [], merge_keys=[])


@pytest.mark.asyncio
async def test_merge_relationships(mock_driver: MagicMock) -> None:
    mock_driver.execute_query.return_value = ([], None, None)
    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        await client.merge_relationships("Person", "p1", "Person", "p2", "KNOWS", [{"p1": 1, "p2": 2}])

    mock_driver.execute_query.assert_awaited_once()
    query = mock_driver.execute_query.await_args[0][0]
    assert "MERGE (source)-[r:`KNOWS`]->(target)" in query


@pytest.mark.asyncio
async def test_to_networkx(mock_driver: MagicMock) -> None:
    # Construct mock graph elements
    node1 = MagicMock(spec=Node)
    node1.element_id = "n1"
    node1.labels = {"Person"}
    node1.items.return_value = [("name", "Alice")]

    node2 = MagicMock(spec=Node)
    node2.element_id = "n2"
    node2.labels = {"Person"}
    node2.items.return_value = [("name", "Bob")]

    rel = MagicMock(spec=Relationship)
    rel.start_node = node1
    rel.end_node = node2
    rel.type = "KNOWS"
    rel.items.return_value = [("since", 2020)]

    # Mock Record
    record = MagicMock()
    record.values.return_value = [node1, rel, node2]

    mock_driver.execute_query.return_value = ([record], None, None)

    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        g = await client.to_networkx("MATCH ...")

    assert isinstance(g, nx.DiGraph)
    assert len(g.nodes) == 2
    assert len(g.edges) == 1
    assert g.nodes["n1"]["name"] == "Alice"
    assert g.edges["n1", "n2"]["since"] == 2020


@pytest.mark.asyncio
async def test_to_networkx_complex_types(mock_driver: MagicMock) -> None:
    # Test Path and List handling
    node = MagicMock(spec=Node)
    node.element_id = "n1"
    node.labels = set()
    node.items.return_value = []

    # Needs a rel for Path
    rel = MagicMock(spec=Relationship)
    rel.start_node = node
    rel.end_node = node
    rel.type = "SELF"
    rel.items.return_value = []

    path = MagicMock(spec=Path)
    path.nodes = [node]
    path.relationships = [rel]  # Added rel to hit line 317

    record = MagicMock()
    record.values.return_value = [[node], path]  # List of nodes, and a Path object

    mock_driver.execute_query.return_value = ([record], None, None)

    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        g = await client.to_networkx("MATCH ...")

    assert "n1" in g.nodes
    assert len(g.edges) == 1


@pytest.mark.asyncio
async def test_to_networkx_failure(mock_driver: MagicMock) -> None:
    mock_driver.execute_query.side_effect = Exception("Graph Error")
    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        with pytest.raises(Exception, match="Graph Error"):
            await client.to_networkx("Q")


@pytest.mark.asyncio
async def test_legacy_node_id_fallback(mock_driver: MagicMock) -> None:
    # Test fallback to `id` if `element_id` is missing (Neo4j 4.x)
    node = MagicMock(spec=Node)
    del node.element_id  # ensure attribute doesn't exist
    node.id = 123
    node.labels = set()
    node.items.return_value = []

    record = MagicMock()
    record.values.return_value = [node]
    mock_driver.execute_query.return_value = ([record], None, None)

    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        g = await client.to_networkx("Q")

    assert 123 in g.nodes


@pytest.mark.asyncio
async def test_to_networkx_unknown_item(mock_driver: MagicMock) -> None:
    # Test unknown item type in record
    record = MagicMock()
    record.values.return_value = ["unknown string"]  # Should be ignored
    mock_driver.execute_query.return_value = ([record], None, None)

    async with Neo4jClient("bolt://host", ("u", "p")) as client:
        g = await client.to_networkx("Q")

    assert len(g.nodes) == 0
