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

import numpy as np
import pytest
from pydantic import ValidationError

from coreason_graph_nexus.link_prediction import LinkPredictor
from coreason_graph_nexus.models import LinkPredictionMethod, LinkPredictionRequest


def test_link_prediction_request_validation() -> None:
    """Test that LinkPredictionRequest validates input correctly."""

    # Valid Request
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.HEURISTIC, heuristic_query="MATCH (a), (b) MERGE (a)-[:LINK]->(b)"
    )
    assert request.method == LinkPredictionMethod.HEURISTIC
    assert request.heuristic_query == "MATCH (a), (b) MERGE (a)-[:LINK]->(b)"

    # Invalid Request: Missing query for Heuristic
    with pytest.raises(ValidationError) as exc:
        LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC)
    assert "heuristic_query is required" in str(exc.value)

    # Valid Request: Semantic
    request_sem = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="A", target_label="B")
    assert request_sem.method == LinkPredictionMethod.SEMANTIC


def test_link_prediction_request_whitespace_validation() -> None:
    """Test that LinkPredictionRequest fails on whitespace-only heuristic query."""

    # Invalid Request: Whitespace query
    with pytest.raises(ValidationError) as exc:
        LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC, heuristic_query="   ")
    assert "cannot be empty/whitespace" in str(exc.value)


def test_heuristic_prediction_execution(mocker: Any) -> None:
    """Test that heuristic prediction executes the cypher query."""
    mock_client = mocker.Mock()
    predictor = LinkPredictor(client=mock_client)

    query = "MATCH (a:Author), (p:Paper) WHERE (a)-[:WROTE]->(p) MERGE (a)-[:EXPERT_ON]->(p)"
    request = LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC, heuristic_query=query)

    predictor.predict_links(request)

    # Verify execute_query was called with the query
    mock_client.execute_query.assert_called_once_with(query)


def test_complex_heuristic_query_execution(mocker: Any) -> None:
    """Test execution of a complex, multi-line Cypher query."""
    mock_client = mocker.Mock()
    predictor = LinkPredictor(client=mock_client)

    query = """
    MATCH (u:User)-[:POSTED]->(p:Post)
    WITH u, count(p) as post_count
    WHERE post_count > 100
    MERGE (u)-[:POWER_USER]->(u)
    """
    request = LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC, heuristic_query=query)

    predictor.predict_links(request)

    # Verify execute_query was called with the exact complex query
    mock_client.execute_query.assert_called_once_with(query)


def test_semantic_prediction_with_unused_query(mocker: Any) -> None:
    """Test semantic prediction request with a heuristic query provided."""
    mock_client = mocker.Mock()
    # Mock execute_query to return empty lists for embeddings
    mock_client.execute_query.return_value = []

    predictor = LinkPredictor(client=mock_client)

    # It is valid to provide heuristic_query even if method is SEMANTIC (Pydantic doesn't forbid it)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC, heuristic_query="MATCH (n) RETURN n", source_label="S", target_label="T"
    )

    # Should run without error (and do nothing because empty embeddings)
    predictor.predict_links(request)

    # Check that execute_query was called to fetch embeddings
    # It might be called once or twice depending on logic.
    # Logic: fetch source, check empty -> return. If source is empty, call_count is 1.
    # In this test, return_value is [], so source fetch is empty.
    # So call_count should be 1.
    assert mock_client.execute_query.call_count == 1


def test_semantic_prediction_validation_failure() -> None:
    """Test that semantic prediction fails if labels are missing."""
    with pytest.raises(ValidationError) as exc:
        LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC)
    assert "source_label and target_label are required" in str(exc.value)


def test_heuristic_execution_failure(mocker: Any) -> None:
    """Test handling of execution failure."""
    mock_client = mocker.Mock()
    mock_client.execute_query.side_effect = Exception("Neo4j Error")

    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC, heuristic_query="INVALID QUERY")

    with pytest.raises(Exception) as exc:
        predictor.predict_links(request)
    assert "Neo4j Error" in str(exc.value)


def test_unknown_method_raises_error(mocker: Any) -> None:
    """Test that unknown method raises NotImplementedError."""
    mock_client = mocker.Mock()
    predictor = LinkPredictor(client=mock_client)

    # Mock request object to bypass Pydantic validation and Enum checks
    request = mocker.Mock()
    # Mock the method as an object with a 'value' attribute
    mock_method = mocker.Mock()
    mock_method.value = "UNKNOWN_METHOD"
    request.method = mock_method
    request.heuristic_query = None

    with pytest.raises(NotImplementedError) as exc:
        predictor.predict_links(request)
    assert "Method" in str(exc.value)
    assert "is not implemented" in str(exc.value)


def test_heuristic_query_missing_defensive_check(mocker: Any) -> None:
    """Test defensive check for missing heuristic query."""
    mock_client = mocker.Mock()
    predictor = LinkPredictor(client=mock_client)

    # Mock request object to bypass Pydantic validation
    request = mocker.Mock()
    request.method = LinkPredictionMethod.HEURISTIC
    request.heuristic_query = None

    with pytest.raises(ValueError) as exc:
        predictor.predict_links(request)
    assert "Heuristic query is missing" in str(exc.value)


def test_semantic_prediction_success(mocker: Any) -> None:
    """Test successful semantic prediction flow."""
    mock_client = mocker.Mock()

    # Mock embeddings
    # Source: 2 nodes
    source_embeddings = [{"id": "s1", "embedding": [1.0, 0.0]}, {"id": "s2", "embedding": [0.0, 1.0]}]
    # Target: 2 nodes
    target_embeddings = [
        {"id": "t1", "embedding": [1.0, 0.0]},  # Similar to s1
        {"id": "t2", "embedding": [0.0, 1.0]},  # Similar to s2
    ]

    # Configure execute_query to return source then target
    mock_client.execute_query.side_effect = [source_embeddings, target_embeddings]

    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC, source_label="Source", target_label="Target", threshold=0.9
    )

    predictor.predict_links(request)

    # Verify execute_query called twice for embeddings
    assert mock_client.execute_query.call_count == 2

    # Verify batch_write called with correct matches
    # s1-t1 (sim=1.0) and s2-t2 (sim=1.0) should be created
    assert mock_client.batch_write.call_count == 1
    call_args = mock_client.batch_write.call_args

    # Check query
    query = call_args[0][0]
    assert "MERGE (source)-[r:`SEMANTIC_LINK`]->(target)" in query

    # Check data
    data = call_args[0][1]
    assert len(data) == 2

    # Verify s1-t1
    pair1 = next((d for d in data if d["start_id"] == "s1"), None)
    assert pair1 is not None
    assert pair1["end_id"] == "t1"
    assert pair1["score"] >= 0.99

    # Verify s2-t2
    pair2 = next((d for d in data if d["start_id"] == "s2"), None)
    assert pair2 is not None
    assert pair2["end_id"] == "t2"


def test_semantic_prediction_no_embeddings(mocker: Any) -> None:
    """Test semantic prediction when no embeddings are found."""
    mock_client = mocker.Mock()
    mock_client.execute_query.return_value = []  # No results

    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="Source", target_label="Target")

    predictor.predict_links(request)

    # Should stop after first fetch returns empty
    # Might be called once or twice depending on logic order, but batch_write should NOT be called
    mock_client.batch_write.assert_not_called()


def test_semantic_prediction_self_loop_exclusion(mocker: Any) -> None:
    """Test that self-loops are excluded in semantic prediction."""
    mock_client = mocker.Mock()

    # Same node in source and target
    embeddings = [{"id": "n1", "embedding": [1.0, 0.0]}]

    mock_client.execute_query.side_effect = [embeddings, embeddings]

    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="Node", target_label="Node")

    predictor.predict_links(request)

    # Similarity will be 1.0, but IDs are same ("n1" == "n1")
    # Should not write anything
    mock_client.batch_write.assert_not_called()


def test_semantic_prediction_no_matches(mocker: Any) -> None:
    """Test semantic prediction where embeddings exist but no pair meets threshold."""
    mock_client = mocker.Mock()

    # Source: 1 node, [1, 0]
    source_embeddings = [{"id": "s1", "embedding": [1.0, 0.0]}]
    # Target: 1 node, [0, 1] (orthogonal, sim=0)
    target_embeddings = [{"id": "t1", "embedding": [0.0, 1.0]}]

    mock_client.execute_query.side_effect = [source_embeddings, target_embeddings]

    predictor = LinkPredictor(client=mock_client)
    # Threshold 0.5 > 0.0
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T", threshold=0.5
    )

    predictor.predict_links(request)

    # Embeddings fetched
    assert mock_client.execute_query.call_count == 2
    # But no relationships written
    mock_client.batch_write.assert_not_called()


def test_semantic_prediction_target_empty(mocker: Any) -> None:
    """Test semantic prediction where source exists but target is empty."""
    mock_client = mocker.Mock()

    # Source: 1 node
    source_embeddings = [{"id": "s1", "embedding": [1.0, 0.0]}]
    # Target: empty
    target_embeddings: list[dict[str, Any]] = []

    mock_client.execute_query.side_effect = [source_embeddings, target_embeddings]

    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T")

    predictor.predict_links(request)

    # Embeddings fetched (both calls)
    assert mock_client.execute_query.call_count == 2
    # No relationships written
    mock_client.batch_write.assert_not_called()


def test_semantic_prediction_missing_labels_defensive(mocker: Any) -> None:
    """Test defensive check for missing labels if Pydantic check bypassed."""
    mock_client = mocker.Mock()
    predictor = LinkPredictor(client=mock_client)

    # Mock request to bypass Pydantic validation
    request = mocker.Mock()
    request.method = LinkPredictionMethod.SEMANTIC
    request.source_label = None
    request.target_label = None

    with pytest.raises(ValueError) as exc:
        predictor.predict_links(request)
    assert "source_label and target_label are required" in str(exc.value)
