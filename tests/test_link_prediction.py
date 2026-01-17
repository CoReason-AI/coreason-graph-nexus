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
from pydantic import ValidationError

from coreason_graph_nexus.link_prediction import LinkPredictor
from coreason_graph_nexus.models import LinkPredictionMethod, LinkPredictionRequest


def test_link_prediction_request_validation() -> None:
    """Test that LinkPredictionRequest validates input correctly."""

    # Valid Request
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.HEURISTIC,
        heuristic_query="MATCH (a), (b) MERGE (a)-[:LINK]->(b)"
    )
    assert request.method == LinkPredictionMethod.HEURISTIC
    assert request.heuristic_query == "MATCH (a), (b) MERGE (a)-[:LINK]->(b)"

    # Invalid Request: Missing query for Heuristic
    with pytest.raises(ValidationError) as exc:
        LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC)
    assert "heuristic_query is required" in str(exc.value)

    # Valid Request: Semantic (query not required yet, though not implemented logic)
    request_sem = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC)
    assert request_sem.method == LinkPredictionMethod.SEMANTIC


def test_heuristic_prediction_execution(mocker) -> None:
    """Test that heuristic prediction executes the cypher query."""
    mock_client = mocker.Mock()
    predictor = LinkPredictor(client=mock_client)

    query = "MATCH (a:Author), (p:Paper) WHERE (a)-[:WROTE]->(p) MERGE (a)-[:EXPERT_ON]->(p)"
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.HEURISTIC,
        heuristic_query=query
    )

    predictor.predict_links(request)

    # Verify execute_query was called with the query
    mock_client.execute_query.assert_called_once_with(query)


def test_semantic_prediction_not_implemented(mocker) -> None:
    """Test that semantic prediction raises NotImplementedError."""
    mock_client = mocker.Mock()
    predictor = LinkPredictor(client=mock_client)

    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC)

    with pytest.raises(NotImplementedError) as exc:
        predictor.predict_links(request)
    assert "Semantic link prediction is not yet implemented" in str(exc.value)


def test_heuristic_execution_failure(mocker) -> None:
    """Test handling of execution failure."""
    mock_client = mocker.Mock()
    mock_client.execute_query.side_effect = Exception("Neo4j Error")

    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.HEURISTIC,
        heuristic_query="INVALID QUERY"
    )

    with pytest.raises(Exception) as exc:
        predictor.predict_links(request)
    assert "Neo4j Error" in str(exc.value)


def test_unknown_method_raises_error(mocker) -> None:
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


def test_heuristic_query_missing_defensive_check(mocker) -> None:
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
