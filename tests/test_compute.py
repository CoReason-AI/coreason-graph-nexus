# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from unittest.mock import MagicMock

import networkx as nx
import pytest

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient
from coreason_graph_nexus.compute import GraphComputer
from coreason_graph_nexus.models import AnalysisAlgo, GraphAnalysisRequest

# --- Fixtures ---


@pytest.fixture
def mock_neo4j_client() -> MagicMock:
    client = MagicMock(spec=Neo4jClient)
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


# --- Tests for GraphAnalysisRequest ---


def test_graph_analysis_request_defaults() -> None:
    req = GraphAnalysisRequest(center_node_id="center1", algorithm=AnalysisAlgo.PAGERANK)
    assert req.depth == 2
    assert req.write_property == "pagerank_score"
    assert req.target_node_id is None


def test_graph_analysis_request_validation_success() -> None:
    req = GraphAnalysisRequest(
        center_node_id="center1",
        target_node_id="target1",
        algorithm=AnalysisAlgo.SHORTEST_PATH,
        depth=3,
        write_property="path",
    )
    assert req.algorithm == "shortest_path"


# --- Tests for GraphComputer ---


def test_fetch_subgraph(mock_neo4j_client: MagicMock) -> None:
    computer = GraphComputer(mock_neo4j_client)
    mock_neo4j_client.to_networkx.return_value = nx.DiGraph()

    computer._fetch_subgraph("nodeA", 3)

    mock_neo4j_client.to_networkx.assert_called_once()
    args, kwargs = mock_neo4j_client.to_networkx.call_args
    query = args[0]
    params = kwargs["parameters"]

    assert "MATCH path = (n)-[*..3]-(m)" in query
    assert "n.id = $center_id" in query
    assert params["center_id"] == "nodeA"


def test_run_analysis_pagerank(mock_neo4j_client: MagicMock, sample_graph: nx.DiGraph) -> None:
    computer = GraphComputer(mock_neo4j_client)
    mock_neo4j_client.to_networkx.return_value = sample_graph

    req = GraphAnalysisRequest(center_node_id="nodeA", algorithm=AnalysisAlgo.PAGERANK, write_property="rank_val")

    result = computer.run_analysis(req)

    # Check if batch_write was called
    mock_neo4j_client.batch_write.assert_called_once()
    args, _ = mock_neo4j_client.batch_write.call_args
    query = args[0]
    data = args[1]

    assert "SET n.`rank_val` = row.value" in query
    assert len(data) == 4  # 4 nodes
    assert "id" in data[0]
    assert "value" in data[0]

    # Check result contains scores
    assert isinstance(result, dict)
    assert "e1" in result


def test_run_analysis_shortest_path(mock_neo4j_client: MagicMock, sample_graph: nx.DiGraph) -> None:
    computer = GraphComputer(mock_neo4j_client)
    mock_neo4j_client.to_networkx.return_value = sample_graph

    req = GraphAnalysisRequest(center_node_id="nodeA", target_node_id="nodeC", algorithm=AnalysisAlgo.SHORTEST_PATH)

    result = computer.run_analysis(req)

    # Should find path: nodeA(e1) -> nodeB(e2) -> nodeC(e3)
    # Result is list of node IDs (internal e1, e2, e3)
    assert result == ["e1", "e2", "e3"]

    # No write back for shortest path
    mock_neo4j_client.batch_write.assert_not_called()


def test_run_analysis_shortest_path_missing_target(mock_neo4j_client: MagicMock) -> None:
    computer = GraphComputer(mock_neo4j_client)
    req = GraphAnalysisRequest(center_node_id="nodeA", algorithm=AnalysisAlgo.SHORTEST_PATH)

    with pytest.raises(ValueError, match="target_node_id is required"):
        computer.run_analysis(req)


def test_run_analysis_louvain(mock_neo4j_client: MagicMock, sample_graph: nx.DiGraph) -> None:
    computer = GraphComputer(mock_neo4j_client)
    mock_neo4j_client.to_networkx.return_value = sample_graph

    req = GraphAnalysisRequest(center_node_id="nodeA", algorithm=AnalysisAlgo.LOUVAIN, write_property="community_id")

    result = computer.run_analysis(req)

    # Check write back
    mock_neo4j_client.batch_write.assert_called_once()
    args, _ = mock_neo4j_client.batch_write.call_args
    query = args[0]

    assert "SET n.`community_id` = row.value" in query
    assert len(result) == 4  # all nodes assigned


def test_run_analysis_empty_graph(mock_neo4j_client: MagicMock) -> None:
    computer = GraphComputer(mock_neo4j_client)
    mock_neo4j_client.to_networkx.return_value = nx.DiGraph()  # Empty

    req = GraphAnalysisRequest(center_node_id="nodeA", algorithm=AnalysisAlgo.PAGERANK)

    result = computer.run_analysis(req)
    assert result == {}
    mock_neo4j_client.batch_write.assert_not_called()


def test_run_analysis_shortest_path_nodes_not_found(mock_neo4j_client: MagicMock, sample_graph: nx.DiGraph) -> None:
    computer = GraphComputer(mock_neo4j_client)
    mock_neo4j_client.to_networkx.return_value = sample_graph

    # Test source not found
    # We call with a mocked fetch that returns a graph WITHOUT nodeX

    G_missing = nx.DiGraph()
    G_missing.add_node("e2", id="nodeB")
    mock_neo4j_client.to_networkx.return_value = G_missing

    with pytest.raises(ValueError, match="Source node .* not found"):
        req_missing = GraphAnalysisRequest(
            center_node_id="nodeX", target_node_id="nodeB", algorithm=AnalysisAlgo.SHORTEST_PATH
        )
        computer.run_analysis(req_missing)

    # Test target not found
    mock_neo4j_client.to_networkx.return_value = sample_graph
    req_missing_target = GraphAnalysisRequest(
        center_node_id="nodeA",  # Exists
        target_node_id="nodeZ",  # Does not exist
        algorithm=AnalysisAlgo.SHORTEST_PATH,
    )
    with pytest.raises(ValueError, match="Target node .* not found"):
        computer.run_analysis(req_missing_target)


def test_run_analysis_shortest_path_no_path(mock_neo4j_client: MagicMock) -> None:
    computer = GraphComputer(mock_neo4j_client)

    # Disjoint graph
    G = nx.DiGraph()
    G.add_node("e1", id="nodeA")
    G.add_node("e2", id="nodeB")
    # No edge

    mock_neo4j_client.to_networkx.return_value = G

    req = GraphAnalysisRequest(center_node_id="nodeA", target_node_id="nodeB", algorithm=AnalysisAlgo.SHORTEST_PATH)

    result = computer.run_analysis(req)
    # Cast to list to satisfy mypy if needed, though assertion handles equality check
    assert result == []


def test_run_analysis_louvain_empty(mock_neo4j_client: MagicMock) -> None:
    computer = GraphComputer(mock_neo4j_client)
    mock_neo4j_client.to_networkx.return_value = nx.DiGraph()

    req = GraphAnalysisRequest(center_node_id="nodeA", algorithm=AnalysisAlgo.LOUVAIN)

    result = computer.run_analysis(req)
    assert result == {}


def test_run_analysis_not_implemented(mock_neo4j_client: MagicMock) -> None:
    computer = GraphComputer(mock_neo4j_client)
    mock_neo4j_client.to_networkx.return_value = nx.DiGraph()

    req = MagicMock(spec=GraphAnalysisRequest)
    req.center_node_id = "a"
    req.depth = 1
    # Mock the algorithm attribute to return something that is not in the Enum
    # (or valid enum but not handled if we had one)
    # We need to simulate that req.algorithm is NOT PAGERANK, SHORTEST_PATH, or LOUVAIN.
    # And when accessed .value, it should act like a string or enum.

    # Let's mock the algorithm field as an object with a value attribute
    mock_algo = MagicMock(spec=AnalysisAlgo)
    # type(mock_algo) needs to have .value access work, but it's a mock so it does.
    # We set the mock to return "unknown" when .value is accessed?
    # No, we set the property .value on the mock instance.
    mock_algo.value = "unknown"
    req.algorithm = mock_algo

    with pytest.raises(NotImplementedError):
        computer.run_analysis(req)


def test_compute_pagerank_cast_any_return(mock_neo4j_client: MagicMock, sample_graph: nx.DiGraph) -> None:
    # This test ensures type casting works, mostly for coverage/sanity
    computer = GraphComputer(mock_neo4j_client)
    res = computer._compute_pagerank(sample_graph, "rank")
    assert isinstance(res, dict)


# --- Complex Scenarios ---


def test_run_analysis_louvain_disconnected(mock_neo4j_client: MagicMock) -> None:
    """Test Louvain on a graph with two disconnected components."""
    G = nx.DiGraph()
    # Component 1
    G.add_node("e1", id="A")
    G.add_node("e2", id="B")
    G.add_edge("e1", "e2")

    # Component 2
    G.add_node("e3", id="C")
    G.add_node("e4", id="D")
    G.add_edge("e3", "e4")

    mock_neo4j_client.to_networkx.return_value = G
    computer = GraphComputer(mock_neo4j_client)

    req = GraphAnalysisRequest(center_node_id="A", algorithm=AnalysisAlgo.LOUVAIN, write_property="comm_id")

    result = computer.run_analysis(req)

    # We expect all 4 nodes to be assigned a community
    assert len(result) == 4
    assert "e1" in result
    assert "e3" in result

    # e1 and e2 should be in same community (or at least valid IDs)
    # e3 and e4 should be in same community
    # The communities might be different or same depending on modularity but usually distinct for disjoint.

    # Verify write back size
    args, _ = mock_neo4j_client.batch_write.call_args
    data = args[1]
    assert len(data) == 4


def test_run_analysis_shortest_path_directionality(mock_neo4j_client: MagicMock) -> None:
    """Test that directionality is respected in shortest path."""
    G = nx.DiGraph()
    G.add_node("e1", id="A")
    G.add_node("e2", id="B")
    G.add_edge("e1", "e2")  # A -> B only

    mock_neo4j_client.to_networkx.return_value = G
    computer = GraphComputer(mock_neo4j_client)

    # Case 1: A -> B should exist
    req1 = GraphAnalysisRequest(center_node_id="A", target_node_id="B", algorithm=AnalysisAlgo.SHORTEST_PATH)
    path1 = computer.run_analysis(req1)
    assert path1 == ["e1", "e2"]

    # Case 2: B -> A should NOT exist
    req2 = GraphAnalysisRequest(center_node_id="B", target_node_id="A", algorithm=AnalysisAlgo.SHORTEST_PATH)
    path2 = computer.run_analysis(req2)
    assert path2 == []


def test_run_analysis_pagerank_sink_node(mock_neo4j_client: MagicMock) -> None:
    """Test PageRank with a sink node (no outgoing edges)."""
    G = nx.DiGraph()
    G.add_node("e1", id="A")
    G.add_node("e2", id="B")
    G.add_edge("e1", "e2")  # A -> B, B is sink

    mock_neo4j_client.to_networkx.return_value = G
    computer = GraphComputer(mock_neo4j_client)

    req = GraphAnalysisRequest(center_node_id="A", algorithm=AnalysisAlgo.PAGERANK)

    scores = computer.run_analysis(req)

    assert len(scores) == 2
    # B should have higher score than A usually (accumlates)
    assert scores["e2"] > scores["e1"]

    # Check values are valid floats
    assert isinstance(scores["e1"], float)
    assert isinstance(scores["e2"], float)
