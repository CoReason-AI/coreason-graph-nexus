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
import pytest
import numpy as np
from coreason_graph_nexus.link_prediction import LinkPredictor
from coreason_graph_nexus.models import LinkPredictionMethod, LinkPredictionRequest

def test_semantic_prediction_idempotency(mocker: Any) -> None:
    """
    Test that semantic prediction generates an idempotent MERGE query.
    """
    mock_client = mocker.Mock()
    # Setup embeddings
    embeddings = [{"id": "1", "embedding": [1.0, 0.0]}]
    # We need separate return values for source and target
    # If source and target are the same list of nodes, we have a self-loop (1->1).
    # The code explicitly skips self-loops: `if s_id == t_id: continue`
    # To test idempotency (and ensuring WRITE happens), we need distinct nodes.

    source_emb = [{"id": "S1", "embedding": [1.0, 0.0]}]
    target_emb = [{"id": "T1", "embedding": [1.0, 0.0]}]

    mock_client.execute_query.side_effect = [source_emb, target_emb]

    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC,
        source_label="S",
        target_label="T",
        threshold=0.9
    )

    predictor.predict_links(request)

    # Verify the query used in batch_write contains MERGE
    assert mock_client.batch_write.called
    query = mock_client.batch_write.call_args[0][0]
    assert "MERGE" in query
    assert "CREATE" not in query # Should prefer MERGE for idempotency

def test_semantic_prediction_disconnected_components(mocker: Any) -> None:
    """
    Test that semantic prediction connects nodes even if they are graph-isolated.

    Scenario:
    - Node A (isolated)
    - Node B (isolated)
    - Embedding Similarity(A, B) > threshold
    - Expect: Link A-B created.
    """
    mock_client = mocker.Mock()

    # Embeddings for Source (A) and Target (B)
    # They are identical vectors -> sim=1.0
    source_emb = [{"id": "A", "embedding": [0.5, 0.5]}]
    target_emb = [{"id": "B", "embedding": [0.5, 0.5]}]

    mock_client.execute_query.side_effect = [source_emb, target_emb]

    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC,
        source_label="S",
        target_label="T",
        threshold=0.9
    )

    predictor.predict_links(request)

    # Verify write
    mock_client.batch_write.assert_called_once()
    data = mock_client.batch_write.call_args[0][1]
    assert len(data) == 1
    assert data[0]["start_id"] == "A"
    assert data[0]["end_id"] == "B"

def test_semantic_prediction_mixed_embedding_types(mocker: Any) -> None:
    """
    Test robustness when embeddings might be lists or numpy arrays (if fetch changed).
    The current code assumes lists from Neo4j, but let's see what happens if we feed tuples.
    """
    mock_client = mocker.Mock()

    source_emb = [{"id": "A", "embedding": (1.0, 0.0)}] # Tuple instead of list
    target_emb = [{"id": "B", "embedding": (1.0, 0.0)}]

    mock_client.execute_query.side_effect = [source_emb, target_emb]

    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC,
        source_label="S",
        target_label="T"
    )

    # np.array can handle tuples, so this should pass without error
    predictor.predict_links(request)

    mock_client.batch_write.assert_called_once()

def test_semantic_prediction_error_handling(mocker: Any) -> None:
    """
    Test that exceptions during embedding fetch are propagated.
    """
    mock_client = mocker.Mock()
    mock_client.execute_query.side_effect = Exception("DB Connection Lost")

    predictor = LinkPredictor(client=mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC,
        source_label="S",
        target_label="T"
    )

    with pytest.raises(Exception) as exc:
        predictor.predict_links(request)
    assert "DB Connection Lost" in str(exc.value)
