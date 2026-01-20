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
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClientAsync
from coreason_graph_nexus.link_prediction import LinkPredictor, LinkPredictorAsync
from coreason_graph_nexus.models import LinkPredictionMethod, LinkPredictionRequest

# ... Sync Tests (Existing) ...


def test_link_prediction_request_validation() -> None:
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.HEURISTIC, heuristic_query="MATCH (a), (b) MERGE (a)-[:LINK]->(b)"
    )
    assert request.method == LinkPredictionMethod.HEURISTIC
    assert request.heuristic_query == "MATCH (a), (b) MERGE (a)-[:LINK]->(b)"

    with pytest.raises(ValidationError) as exc:
        LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC)
    assert "heuristic_query is required" in str(exc.value)

    request_sem = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="A", target_label="B")
    assert request_sem.method == LinkPredictionMethod.SEMANTIC


def test_link_prediction_request_whitespace_validation() -> None:
    with pytest.raises(ValidationError) as exc:
        LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC, heuristic_query="   ")
    assert "cannot be empty/whitespace" in str(exc.value)


def test_heuristic_prediction_execution(mocker: Any) -> None:
    mock_client = mocker.Mock()
    predictor = LinkPredictor(client=mock_client)
    query = "MATCH (a:Author), (p:Paper) WHERE (a)-[:WROTE]->(p) MERGE (a)-[:EXPERT_ON]->(p)"
    request = LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC, heuristic_query=query)
    predictor.predict_links(request)
    mock_client.execute_query.assert_called_once_with(query)


def test_complex_heuristic_query_execution(mocker: Any) -> None:
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
    mock_client.execute_query.assert_called_once_with(query)


def test_semantic_prediction_with_unused_query(mocker: Any) -> None:
    mock_client = mocker.Mock()
    mock_client.execute_query.return_value = []
    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC, heuristic_query="MATCH (n) RETURN n", source_label="S", target_label="T"
    )
    predictor.predict_links(request)
    assert mock_client.execute_query.call_count == 1


def test_semantic_prediction_validation_failure() -> None:
    with pytest.raises(ValidationError) as exc:
        LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC)
    assert "source_label and target_label are required" in str(exc.value)


def test_heuristic_execution_failure(mocker: Any) -> None:
    mock_client = mocker.Mock()
    mock_client.execute_query.side_effect = Exception("Neo4j Error")
    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC, heuristic_query="INVALID QUERY")
    with pytest.raises(Exception) as exc:
        predictor.predict_links(request)
    assert "Neo4j Error" in str(exc.value)


def test_unknown_method_raises_error(mocker: Any) -> None:
    mock_client = mocker.Mock()
    predictor = LinkPredictor(client=mock_client)
    request = mocker.Mock()
    mock_method = mocker.Mock()
    mock_method.value = "UNKNOWN_METHOD"
    request.method = mock_method
    request.heuristic_query = None
    with pytest.raises(NotImplementedError) as exc:
        predictor.predict_links(request)
    assert "Method" in str(exc.value)


def test_heuristic_query_missing_defensive_check(mocker: Any) -> None:
    mock_client = mocker.Mock()
    predictor = LinkPredictor(client=mock_client)
    request = mocker.Mock()
    request.method = LinkPredictionMethod.HEURISTIC
    request.heuristic_query = None
    with pytest.raises(ValueError) as exc:
        predictor.predict_links(request)
    assert "Heuristic query is missing" in str(exc.value)


def test_semantic_prediction_success(mocker: Any) -> None:
    mock_client = mocker.Mock()
    source_embeddings = [{"id": "s1", "embedding": [1.0, 0.0]}, {"id": "s2", "embedding": [0.0, 1.0]}]
    target_embeddings = [
        {"id": "t1", "embedding": [1.0, 0.0]},
        {"id": "t2", "embedding": [0.0, 1.0]},
    ]
    mock_client.execute_query.side_effect = [source_embeddings, target_embeddings]
    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC, source_label="Source", target_label="Target", threshold=0.9
    )
    predictor.predict_links(request)
    assert mock_client.execute_query.call_count == 2
    assert mock_client.batch_write.call_count == 1
    call_args = mock_client.batch_write.call_args
    data = call_args[0][1]
    assert len(data) == 2


def test_semantic_prediction_no_embeddings(mocker: Any) -> None:
    mock_client = mocker.Mock()
    mock_client.execute_query.return_value = []
    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="Source", target_label="Target")
    predictor.predict_links(request)
    mock_client.batch_write.assert_not_called()


def test_semantic_prediction_self_loop_exclusion(mocker: Any) -> None:
    mock_client = mocker.Mock()
    embeddings = [{"id": "n1", "embedding": [1.0, 0.0]}]
    mock_client.execute_query.side_effect = [embeddings, embeddings]
    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="Node", target_label="Node")
    predictor.predict_links(request)
    mock_client.batch_write.assert_not_called()


def test_semantic_prediction_no_matches(mocker: Any) -> None:
    mock_client = mocker.Mock()
    source_embeddings = [{"id": "s1", "embedding": [1.0, 0.0]}]
    target_embeddings = [{"id": "t1", "embedding": [0.0, 1.0]}]
    mock_client.execute_query.side_effect = [source_embeddings, target_embeddings]
    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T", threshold=0.5
    )
    predictor.predict_links(request)
    assert mock_client.execute_query.call_count == 2
    mock_client.batch_write.assert_not_called()


def test_semantic_prediction_target_empty(mocker: Any) -> None:
    mock_client = mocker.Mock()
    source_embeddings = [{"id": "s1", "embedding": [1.0, 0.0]}]
    target_embeddings: list[dict[str, Any]] = []
    mock_client.execute_query.side_effect = [source_embeddings, target_embeddings]
    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T")
    predictor.predict_links(request)
    assert mock_client.execute_query.call_count == 2
    mock_client.batch_write.assert_not_called()


def test_semantic_prediction_missing_labels_defensive(mocker: Any) -> None:
    mock_client = mocker.Mock()
    predictor = LinkPredictor(client=mock_client)
    request = mocker.Mock()
    request.method = LinkPredictionMethod.SEMANTIC
    request.source_label = None
    request.target_label = None
    with pytest.raises(ValueError) as exc:
        predictor.predict_links(request)
    assert "source_label and target_label are required" in str(exc.value)


def test_semantic_prediction_zero_vector(mocker: Any) -> None:
    mock_client = mocker.Mock()
    source_embeddings = [{"id": "s1", "embedding": [0.0, 0.0]}]
    target_embeddings = [{"id": "t1", "embedding": [1.0, 0.0]}]
    mock_client.execute_query.side_effect = [source_embeddings, target_embeddings]
    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T")
    predictor.predict_links(request)
    mock_client.batch_write.assert_not_called()


def test_semantic_prediction_dimension_mismatch(mocker: Any) -> None:
    mock_client = mocker.Mock()
    source_embeddings = [{"id": "s1", "embedding": [1.0, 0.0]}]
    target_embeddings = [{"id": "t1", "embedding": [1.0, 0.0, 1.0]}]
    mock_client.execute_query.side_effect = [source_embeddings, target_embeddings]
    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T")
    with pytest.raises(ValueError) as exc:
        predictor.predict_links(request)
    assert "Incompatible dimension" in str(exc.value) or "shapes" in str(exc.value)


def test_semantic_prediction_threshold_boundary(mocker: Any) -> None:
    mock_client = mocker.Mock()
    source_embeddings = [{"id": "s1", "embedding": [1.0, 0.0]}]
    target_embeddings = [{"id": "t1", "embedding": [1.0, 0.0]}]
    mock_client.execute_query.side_effect = [source_embeddings, target_embeddings]
    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T", threshold=1.0
    )
    predictor.predict_links(request)
    mock_client.batch_write.assert_called_once()


def test_semantic_prediction_invalid_embedding_format(mocker: Any) -> None:
    mock_client = mocker.Mock()
    source_embeddings = [{"id": "s1", "embedding": "invalid-string"}]
    target_embeddings = [{"id": "t1", "embedding": [1.0, 0.0]}]
    mock_client.execute_query.side_effect = [source_embeddings, target_embeddings]
    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T")
    with pytest.raises((ValueError, TypeError)):
        predictor.predict_links(request)


# ... Async Tests ...


@pytest.mark.asyncio
async def test_link_prediction_async_heuristic(mocker: Any) -> None:
    """Test async heuristic link prediction."""
    mock_client = mocker.Mock(spec=Neo4jClientAsync)
    mock_client.execute_query = AsyncMock(return_value=[])

    predictor = LinkPredictorAsync(client=mock_client)

    request = LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC, heuristic_query="MATCH (n) RETURN n")

    await predictor.predict_links(request)
    mock_client.execute_query.assert_called_once()


@pytest.mark.asyncio
async def test_link_prediction_async_heuristic_failure(mocker: Any) -> None:
    mock_client = mocker.Mock(spec=Neo4jClientAsync)
    mock_client.execute_query = AsyncMock(side_effect=Exception("Async Fail"))

    predictor = LinkPredictorAsync(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC, heuristic_query="Q")

    # We want to verify logger call too if possible, but assert exception propagation
    with pytest.raises(Exception, match="Async Fail"):
        await predictor.predict_links(request)


@pytest.mark.asyncio
async def test_semantic_prediction_async(mocker: Any) -> None:
    """Test async semantic prediction with threaded compute."""
    mock_client = mocker.Mock(spec=Neo4jClientAsync)

    source_embeddings = [{"id": "s1", "embedding": [1.0, 0.0]}]
    target_embeddings = [{"id": "t1", "embedding": [1.0, 0.0]}]

    mock_client.execute_query = AsyncMock(side_effect=[source_embeddings, target_embeddings])
    mock_client.batch_write = AsyncMock()

    predictor = LinkPredictorAsync(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T", threshold=0.9
    )

    await predictor.predict_links(request)

    assert mock_client.execute_query.call_count == 2
    mock_client.batch_write.assert_called_once()


@pytest.mark.asyncio
async def test_semantic_prediction_async_no_embeddings(mocker: Any) -> None:
    mock_client = mocker.Mock(spec=Neo4jClientAsync)
    mock_client.execute_query = AsyncMock(return_value=[])

    predictor = LinkPredictorAsync(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T")

    await predictor.predict_links(request)
    mock_client.execute_query.assert_called_once()


@pytest.mark.asyncio
async def test_semantic_prediction_async_no_target_embeddings(mocker: Any) -> None:
    """Test async semantic prediction where source exists but target is empty."""
    mock_client = mocker.Mock(spec=Neo4jClientAsync)

    source_embeddings = [{"id": "s1", "embedding": [1.0, 0.0]}]
    # First call returns source, second call (target) returns empty
    mock_client.execute_query = AsyncMock(side_effect=[source_embeddings, []])

    predictor = LinkPredictorAsync(client=mock_client)
    request = LinkPredictionRequest(method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T")

    await predictor.predict_links(request)

    assert mock_client.execute_query.call_count == 2
    mock_client.batch_write.assert_not_called()


@pytest.mark.asyncio
async def test_semantic_prediction_async_no_matches(mocker: Any) -> None:
    """Test async semantic prediction where embeddings exist but no matches found."""
    mock_client = mocker.Mock(spec=Neo4jClientAsync)

    source_embeddings = [{"id": "s1", "embedding": [1.0, 0.0]}]
    target_embeddings = [{"id": "t1", "embedding": [0.0, 1.0]}]
    mock_client.execute_query = AsyncMock(side_effect=[source_embeddings, target_embeddings])

    predictor = LinkPredictorAsync(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC, source_label="S", target_label="T", threshold=0.9
    )

    await predictor.predict_links(request)

    assert mock_client.execute_query.call_count == 2
    mock_client.batch_write.assert_not_called()


@pytest.mark.asyncio
async def test_link_prediction_async_unknown_method(mocker: Any) -> None:
    mock_client = mocker.Mock(spec=Neo4jClientAsync)
    predictor = LinkPredictorAsync(client=mock_client)

    request = mocker.Mock()
    mock_method = mocker.Mock()
    mock_method.value = "UNKNOWN"
    request.method = mock_method
    request.heuristic_query = None

    with pytest.raises(NotImplementedError):
        await predictor.predict_links(request)


@pytest.mark.asyncio
async def test_link_prediction_async_missing_query(mocker: Any) -> None:
    mock_client = mocker.Mock(spec=Neo4jClientAsync)
    predictor = LinkPredictorAsync(client=mock_client)

    request = mocker.Mock()
    request.method = LinkPredictionMethod.HEURISTIC
    request.heuristic_query = None

    with pytest.raises(ValueError):
        await predictor.predict_links(request)


@pytest.mark.asyncio
async def test_semantic_prediction_async_missing_labels(mocker: Any) -> None:
    mock_client = mocker.Mock(spec=Neo4jClientAsync)
    predictor = LinkPredictorAsync(client=mock_client)

    request = mocker.Mock()
    request.method = LinkPredictionMethod.SEMANTIC
    request.source_label = None
    request.target_label = None

    with pytest.raises(ValueError, match="source_label and target_label are required"):
        await predictor.predict_links(request)
