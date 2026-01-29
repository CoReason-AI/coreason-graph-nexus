# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from unittest.mock import AsyncMock, MagicMock

import networkx as nx
import pytest

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClientAsync
from coreason_graph_nexus.compute import GraphComputerAsync
from coreason_graph_nexus.models import AnalysisAlgo, GraphAnalysisRequest

# --- Fixtures ---


@pytest.fixture
def mock_client_async() -> MagicMock:
    client = MagicMock(spec=Neo4jClientAsync)
    client.to_networkx = AsyncMock()
    client.batch_write = AsyncMock()
    return client


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    """Creates a sample graph with 4 nodes and some edges."""
    G = nx.DiGraph()
    # Nodes with 'id' property (business ID) and some are keyed by element_id (internal)
    G.add_node("e1", id="nodeA", labels=["Person"])
    G.add_node("e2", id="nodeB", labels=["Person"])
    G.add_node("e3", id="nodeC", labels=["Person"])
    G.add_node("e4", id="nodeD", labels=["Person"])

    G.add_edge("e1", "e2")
    G.add_edge("e2", "e3")
    G.add_edge("e1", "e4")
    return G


# --- Tests for GraphComputerAsync ---


@pytest.mark.asyncio
async def test_run_analysis_pagerank(mock_client_async: MagicMock, sample_graph: nx.DiGraph) -> None:
    computer = GraphComputerAsync(mock_client_async)
    mock_client_async.to_networkx.return_value = sample_graph

    req = GraphAnalysisRequest(center_node_id="nodeA", algorithm=AnalysisAlgo.PAGERANK, write_property="rank_val")

    result = await computer.run_analysis(req)

    # Check if batch_write was called
    mock_client_async.batch_write.assert_called_once()
    args, _ = mock_client_async.batch_write.call_args
    query = args[0]
    data = args[1]

    assert "SET n.`rank_val` = row.value" in query
    assert len(data) == 4  # 4 nodes
    assert "id" in data[0]
    assert "value" in data[0]

    # Check result contains scores
    assert isinstance(result, dict)
    assert "e1" in result


@pytest.mark.asyncio
async def test_run_analysis_shortest_path(mock_client_async: MagicMock, sample_graph: nx.DiGraph) -> None:
    computer = GraphComputerAsync(mock_client_async)
    mock_client_async.to_networkx.return_value = sample_graph

    req = GraphAnalysisRequest(center_node_id="nodeA", target_node_id="nodeC", algorithm=AnalysisAlgo.SHORTEST_PATH)

    result = await computer.run_analysis(req)

    # Should find path: nodeA(e1) -> nodeB(e2) -> nodeC(e3)
    # Result is list of node IDs (internal e1, e2, e3)
    assert result == ["e1", "e2", "e3"]

    # No write back for shortest path
    mock_client_async.batch_write.assert_not_called()


@pytest.mark.asyncio
async def test_run_analysis_shortest_path_missing_target(mock_client_async: MagicMock) -> None:
    computer = GraphComputerAsync(mock_client_async)
    req = GraphAnalysisRequest(center_node_id="nodeA", algorithm=AnalysisAlgo.SHORTEST_PATH)

    with pytest.raises(ValueError, match="target_node_id is required"):
        await computer.run_analysis(req)


@pytest.mark.asyncio
async def test_run_analysis_louvain(mock_client_async: MagicMock, sample_graph: nx.DiGraph) -> None:
    computer = GraphComputerAsync(mock_client_async)
    mock_client_async.to_networkx.return_value = sample_graph

    req = GraphAnalysisRequest(center_node_id="nodeA", algorithm=AnalysisAlgo.LOUVAIN, write_property="community_id")

    result = await computer.run_analysis(req)

    # Check write back
    mock_client_async.batch_write.assert_called_once()
    args, _ = mock_client_async.batch_write.call_args
    query = args[0]

    assert "SET n.`community_id` = row.value" in query
    assert len(result) == 4  # all nodes assigned


@pytest.mark.asyncio
async def test_run_analysis_empty_graph(mock_client_async: MagicMock) -> None:
    computer = GraphComputerAsync(mock_client_async)
    mock_client_async.to_networkx.return_value = nx.DiGraph()  # Empty

    req = GraphAnalysisRequest(center_node_id="nodeA", algorithm=AnalysisAlgo.PAGERANK)

    result = await computer.run_analysis(req)
    assert result == {}
    mock_client_async.batch_write.assert_not_called()


@pytest.mark.asyncio
async def test_run_analysis_shortest_path_nodes_not_found(
    mock_client_async: MagicMock, sample_graph: nx.DiGraph
) -> None:
    computer = GraphComputerAsync(mock_client_async)
    mock_client_async.to_networkx.return_value = sample_graph

    # Test source not found
    # We call with a mocked fetch that returns a graph WITHOUT nodeX

    G_missing = nx.DiGraph()
    G_missing.add_node("e2", id="nodeB")
    mock_client_async.to_networkx.return_value = G_missing

    with pytest.raises(ValueError, match="Source node .* not found"):
        req_missing = GraphAnalysisRequest(
            center_node_id="nodeX", target_node_id="nodeB", algorithm=AnalysisAlgo.SHORTEST_PATH
        )
        await computer.run_analysis(req_missing)

    # Test target not found
    mock_client_async.to_networkx.return_value = sample_graph
    req_missing_target = GraphAnalysisRequest(
        center_node_id="nodeA",  # Exists
        target_node_id="nodeZ",  # Does not exist
        algorithm=AnalysisAlgo.SHORTEST_PATH,
    )
    with pytest.raises(ValueError, match="Target node .* not found"):
        await computer.run_analysis(req_missing_target)


@pytest.mark.asyncio
async def test_run_analysis_shortest_path_no_path(mock_client_async: MagicMock) -> None:
    computer = GraphComputerAsync(mock_client_async)

    # Disjoint graph
    G = nx.DiGraph()
    G.add_node("e1", id="nodeA")
    G.add_node("e2", id="nodeB")
    # No edge

    mock_client_async.to_networkx.return_value = G

    req = GraphAnalysisRequest(center_node_id="nodeA", target_node_id="nodeB", algorithm=AnalysisAlgo.SHORTEST_PATH)

    result = await computer.run_analysis(req)
    assert result == []


@pytest.mark.asyncio
async def test_run_analysis_louvain_empty(mock_client_async: MagicMock) -> None:
    computer = GraphComputerAsync(mock_client_async)
    mock_client_async.to_networkx.return_value = nx.DiGraph()

    req = GraphAnalysisRequest(center_node_id="nodeA", algorithm=AnalysisAlgo.LOUVAIN)

    result = await computer.run_analysis(req)
    assert result == {}


@pytest.mark.asyncio
async def test_run_analysis_not_implemented(mock_client_async: MagicMock) -> None:
    computer = GraphComputerAsync(mock_client_async)
    mock_client_async.to_networkx.return_value = nx.DiGraph()

    req = MagicMock(spec=GraphAnalysisRequest)
    req.center_node_id = "a"
    req.depth = 1
    mock_algo = MagicMock(spec=AnalysisAlgo)
    mock_algo.value = "unknown"
    req.algorithm = mock_algo

    with pytest.raises(NotImplementedError):
        await computer.run_analysis(req)
