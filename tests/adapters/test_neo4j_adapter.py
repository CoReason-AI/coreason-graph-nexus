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
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from neo4j.exceptions import ServiceUnavailable

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient


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


# --- New Tests for Edge Cases & Complex Scenarios ---


def test_execute_query_empty_result(mock_driver: MagicMock) -> None:
    """Test successful query execution returning no records."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.return_value = ([], None, None)

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
    result = client.execute_query("MATCH (n:NonExistent) RETURN n")

    assert result == []


def test_execute_query_complex_data(mock_driver: MagicMock) -> None:
    """Test handling of complex return types (nested dicts, lists, nulls)."""
    driver_instance = mock_driver.return_value

    complex_data: dict[str, Any] = {
        "id": 1,
        "labels": ["Person", "Employee"],
        "metadata": {"source": "etl", "active": True},
        "reports_to": None,
    }

    mock_record = MagicMock()
    mock_record.data.return_value = complex_data
    driver_instance.execute_query.return_value = ([mock_record], None, None)

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
    result = client.execute_query("MATCH (p:Person) RETURN p")

    assert len(result) == 1
    assert result[0] == complex_data


def test_execute_query_complex_params(mock_driver: MagicMock) -> None:
    """Test passing complex parameters (lists, dicts) to the query."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.return_value = ([], None, None)

    params = {
        "names": ["Alice", "Bob"],
        "props": {"department": "Engineering"},
        "limit": 10,
    }

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
    client.execute_query("MATCH (n) WHERE n.name IN $names", params)

    driver_instance.execute_query.assert_called_once_with(
        "MATCH (n) WHERE n.name IN $names",
        parameters_=params,
        database_="neo4j",
    )


def test_custom_database_selection(mock_driver: MagicMock) -> None:
    """Test that queries are executed against the specified custom database."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.return_value = ([], None, None)

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"), database="analytics")
    client.execute_query("MATCH (n) RETURN count(n)")

    driver_instance.execute_query.assert_called_once()
    call_args = driver_instance.execute_query.call_args
    assert call_args.kwargs["database_"] == "analytics"


def test_connection_failure_during_query(mock_driver: MagicMock) -> None:
    """Test handling of ServiceUnavailable raised during query execution."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.side_effect = ServiceUnavailable("Connection dropped")

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    with pytest.raises(ServiceUnavailable, match="Connection dropped"):
        client.execute_query("MATCH (n) RETURN n")


# --- Tests for to_networkx ---


class MockNode:
    """Dummy class to replace neo4j.graph.Node in tests."""

    def __init__(self, element_id: str, labels: list[str], properties: dict[str, Any]) -> None:
        self.element_id = element_id
        self.labels = labels
        self.properties = properties

    def items(self) -> Any:
        return self.properties.items()


class MockRelationship:
    """Dummy class to replace neo4j.graph.Relationship in tests."""

    def __init__(
        self, element_id: str, type_: str, start_node: MockNode, end_node: MockNode, properties: dict[str, Any]
    ) -> None:
        self.element_id = element_id
        self.type = type_
        self.start_node = start_node
        self.end_node = end_node
        self.properties = properties

    def items(self) -> Any:
        return self.properties.items()


class MockPath:
    """Dummy class to replace neo4j.graph.Path in tests."""

    def __init__(self, nodes: list[MockNode], relationships: list[MockRelationship]) -> None:
        self.nodes = nodes
        self.relationships = relationships


def test_to_networkx_nodes_and_relationships(mock_driver: MagicMock) -> None:
    """Test converting Neo4j results to NetworkX graph."""
    driver_instance = mock_driver.return_value

    # Patch the types in the module so isinstance() works
    with (
        patch("coreason_graph_nexus.adapters.neo4j_adapter.Node", new=MockNode),
        patch("coreason_graph_nexus.adapters.neo4j_adapter.Relationship", new=MockRelationship),
    ):
        node_a = MockNode("node1", ["Person"], {"name": "Alice"})
        node_b = MockNode("node2", ["Person"], {"name": "Bob"})
        rel = MockRelationship("rel1", "KNOWS", node_a, node_b, {"since": 2022})

        mock_record = MagicMock()
        mock_record.values.return_value = [node_a, rel, node_b]

        driver_instance.execute_query.return_value = ([mock_record], None, None)

        client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
        g = client.to_networkx("MATCH (a)-[r]->(b) RETURN a, r, b")

        assert isinstance(g, nx.DiGraph)
        assert len(g.nodes) == 2
        assert len(g.edges) == 1

        assert g.nodes["node1"]["name"] == "Alice"
        assert g.nodes["node1"]["labels"] == ["Person"]

        assert g.nodes["node2"]["name"] == "Bob"

        assert g.has_edge("node1", "node2")
        assert g.edges["node1", "node2"]["type"] == "KNOWS"
        assert g.edges["node1", "node2"]["since"] == 2022


def test_to_networkx_path(mock_driver: MagicMock) -> None:
    """Test converting a Neo4j Path to NetworkX."""
    driver_instance = mock_driver.return_value

    with (
        patch("coreason_graph_nexus.adapters.neo4j_adapter.Node", new=MockNode),
        patch("coreason_graph_nexus.adapters.neo4j_adapter.Relationship", new=MockRelationship),
        patch("coreason_graph_nexus.adapters.neo4j_adapter.Path", new=MockPath),
    ):
        node_a = MockNode("n1", ["A"], {})
        node_b = MockNode("n2", ["B"], {})
        rel = MockRelationship("r1", "LINK", node_a, node_b, {})
        path = MockPath([node_a, node_b], [rel])

        mock_record = MagicMock()
        mock_record.values.return_value = [path]
        driver_instance.execute_query.return_value = ([mock_record], None, None)

        client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
        g = client.to_networkx("MATCH p=(a)-[]-(b) RETURN p")

        assert len(g.nodes) == 2
        assert len(g.edges) == 1
        assert g.has_edge("n1", "n2")


def test_to_networkx_list_of_nodes(mock_driver: MagicMock) -> None:
    """Test converting a list of nodes (e.g., collect(n)) to NetworkX."""
    driver_instance = mock_driver.return_value

    with patch("coreason_graph_nexus.adapters.neo4j_adapter.Node", new=MockNode):
        node = MockNode("n1", [], {})

        mock_record = MagicMock()
        mock_record.values.return_value = [[node]]  # A list containing a node
        driver_instance.execute_query.return_value = ([mock_record], None, None)

        client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
        g = client.to_networkx("MATCH (n) RETURN collect(n)")

        assert len(g.nodes) == 1
        assert "n1" in g.nodes


def test_to_networkx_failure(mock_driver: MagicMock) -> None:
    """Test failure handling in to_networkx."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.side_effect = Exception("DB Error")

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    with pytest.raises(Exception, match="DB Error"):
        client.to_networkx("MATCH (n) RETURN n")


# --- Additional Edge Cases & Complex Scenarios ---


def test_to_networkx_mixed_content_and_nulls(mock_driver: MagicMock) -> None:
    """Test converting mixed content (nodes, scalars, None) where scalars/None should be ignored."""
    driver_instance = mock_driver.return_value

    with (
        patch("coreason_graph_nexus.adapters.neo4j_adapter.Node", new=MockNode),
        patch("coreason_graph_nexus.adapters.neo4j_adapter.Relationship", new=MockRelationship),
    ):
        node = MockNode("n1", ["Label"], {"p": 1})
        rel = MockRelationship("r1", "REL", node, node, {})  # Self loop for convenience

        # Row contains: Node, Integer, None, String, Relationship
        mock_record = MagicMock()
        mock_record.values.return_value = [node, 123, None, "test", rel]

        driver_instance.execute_query.return_value = ([mock_record], None, None)

        client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
        g = client.to_networkx("MATCH (n)-[r]-(m) RETURN n, count(*), r")

        assert len(g.nodes) == 1
        assert len(g.edges) == 1
        assert "n1" in g.nodes
        assert g.has_edge("n1", "n1")


def test_to_networkx_cycles_and_self_loops(mock_driver: MagicMock) -> None:
    """Test handling of cycles (A->B->A) and self-loops (A->A)."""
    driver_instance = mock_driver.return_value

    with (
        patch("coreason_graph_nexus.adapters.neo4j_adapter.Node", new=MockNode),
        patch("coreason_graph_nexus.adapters.neo4j_adapter.Relationship", new=MockRelationship),
    ):
        # A <-> B Cycle
        node_a = MockNode("A", [], {})
        node_b = MockNode("B", [], {})
        rel_ab = MockRelationship("r1", "TO", node_a, node_b, {})
        rel_ba = MockRelationship("r2", "BACK", node_b, node_a, {})

        # Self loop C -> C
        node_c = MockNode("C", [], {})
        rel_cc = MockRelationship("r3", "SELF", node_c, node_c, {})

        mock_record_1 = MagicMock()
        mock_record_1.values.return_value = [node_a, rel_ab, node_b]

        mock_record_2 = MagicMock()
        mock_record_2.values.return_value = [node_b, rel_ba, node_a]

        mock_record_3 = MagicMock()
        mock_record_3.values.return_value = [node_c, rel_cc]

        driver_instance.execute_query.return_value = ([mock_record_1, mock_record_2, mock_record_3], None, None)

        client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
        g = client.to_networkx("MATCH paths")

        # Verify A <-> B
        assert g.has_edge("A", "B")
        assert g.has_edge("B", "A")
        assert g.edges["A", "B"]["type"] == "TO"
        assert g.edges["B", "A"]["type"] == "BACK"

        # Verify C -> C
        assert g.has_edge("C", "C")
        assert g.edges["C", "C"]["type"] == "SELF"


def test_to_networkx_idempotency_and_updates(mock_driver: MagicMock) -> None:
    """Test that later occurrences of a node in the stream update its properties."""
    driver_instance = mock_driver.return_value

    with patch("coreason_graph_nexus.adapters.neo4j_adapter.Node", new=MockNode):
        # First occurrence: property 'a'=1
        node_v1 = MockNode("n1", ["L"], {"a": 1})
        # Second occurrence: property 'b'=2 (simulating merge or updated state in stream)
        node_v2 = MockNode("n1", ["L"], {"b": 2})

        rec1 = MagicMock()
        rec1.values.return_value = [node_v1]
        rec2 = MagicMock()
        rec2.values.return_value = [node_v2]

        driver_instance.execute_query.return_value = ([rec1, rec2], None, None)

        client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
        g = client.to_networkx("...")

        assert len(g.nodes) == 1
        assert "n1" in g.nodes
        # NetworkX should merge attributes if add_node is called twice
        # Wait, if I call add_node(id, attr=val), it updates/adds that attr.
        # But if the second call does NOT have 'a', does it remove 'a'? No.
        # So expected result is {'a': 1, 'b': 2, 'labels': ['L']}

        attrs = g.nodes["n1"]
        assert attrs["a"] == 1
        assert attrs["b"] == 2
        assert attrs["labels"] == ["L"]


def test_to_networkx_list_of_mixed_types(mock_driver: MagicMock) -> None:
    """Test converting a list of mixed types (Nodes, Relationships, others)."""
    driver_instance = mock_driver.return_value

    with (
        patch("coreason_graph_nexus.adapters.neo4j_adapter.Node", new=MockNode),
        patch("coreason_graph_nexus.adapters.neo4j_adapter.Relationship", new=MockRelationship),
    ):
        node = MockNode("n1", [], {})
        # Note: Relationship requires start/end nodes in the constructor
        rel = MockRelationship("r1", "REL", node, node, {})

        # List containing a node, a relationship, and an integer (which should be ignored)
        mixed_list = [node, rel, 123]

        mock_record = MagicMock()
        mock_record.values.return_value = [mixed_list]
        driver_instance.execute_query.return_value = ([mock_record], None, None)

        client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
        g = client.to_networkx("MATCH (n)-[r]-(m) RETURN collect([n, r])")

        assert len(g.nodes) == 1
        assert "n1" in g.nodes
        assert g.has_edge("n1", "n1")
