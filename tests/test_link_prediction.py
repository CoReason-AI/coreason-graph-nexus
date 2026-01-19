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

import pytest
from pytest_mock import MockerFixture

from coreason_graph_nexus.link_prediction import LinkPredictor
from coreason_graph_nexus.models import LinkPredictionMethod, LinkPredictionRequest


@pytest.fixture
def mock_client(mocker: MockerFixture) -> MagicMock:
    client = mocker.Mock()
    client.execute_query = mocker.AsyncMock()
    client.batch_write = mocker.AsyncMock()
    return cast(MagicMock, client)


@pytest.mark.asyncio
async def test_heuristic_prediction(mock_client: MagicMock) -> None:
    predictor = LinkPredictor(mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.HEURISTIC,
        heuristic_query="MATCH (a)-[:X]->(b) MERGE (a)-[:Y]->(b)",
    )

    await predictor.predict_links(request)
    mock_client.execute_query.assert_awaited_once_with(request.heuristic_query)


@pytest.mark.asyncio
async def test_heuristic_prediction_missing_query(mock_client: MagicMock) -> None:
    predictor = LinkPredictor(mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.HEURISTIC,
        heuristic_query="temp",  # satisfy validator
    )
    request.heuristic_query = ""  # Manually clear it to test method logic

    with pytest.raises(ValueError, match="Heuristic query is missing"):
        await predictor.predict_links(request)


@pytest.mark.asyncio
async def test_semantic_prediction(mock_client: MagicMock) -> None:
    predictor = LinkPredictor(mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC,
        source_label="Source",
        target_label="Target",
        threshold=0.5,
    )

    # Mock embeddings
    mock_client.execute_query.side_effect = [
        [{"id": "1", "embedding": [1.0, 0.0]}, {"id": "2", "embedding": [0.0, 1.0]}],  # Source
        [{"id": "3", "embedding": [1.0, 0.0]}, {"id": "4", "embedding": [0.0, 1.0]}],  # Target
    ]

    await predictor.predict_links(request)

    assert mock_client.execute_query.await_count == 2
    mock_client.batch_write.assert_awaited_once()

    # Check data passed to batch_write
    call_args = mock_client.batch_write.call_args
    query, data = call_args[0]
    assert "UNWIND $batch" in query
    assert len(data) == 2
    assert {"start_id": "1", "end_id": "3", "score": 1.0} in data
    assert {"start_id": "2", "end_id": "4", "score": 1.0} in data


@pytest.mark.asyncio
async def test_semantic_prediction_no_embeddings(mock_client: MagicMock) -> None:
    predictor = LinkPredictor(mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC,
        source_label="Source",
        target_label="Target",
    )

    # First call returns empty
    mock_client.execute_query.return_value = []

    await predictor.predict_links(request)

    # Should stop after first empty fetch
    assert mock_client.execute_query.await_count == 1
    mock_client.batch_write.assert_not_awaited()


@pytest.mark.asyncio
async def test_not_implemented_method(mock_client: MagicMock) -> None:
    predictor = LinkPredictor(mock_client)
    # Mock request object to bypass Enum validation and .value access failure
    request = MagicMock()
    request.method.value = "INVALID"

    with pytest.raises(NotImplementedError):
        await predictor.predict_links(request)
