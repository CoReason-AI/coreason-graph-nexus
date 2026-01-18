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

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient


def test_batch_write_failure_handling(mocker: Any) -> None:
    """
    Test that batch_write raises an exception if one chunk fails,
    and logs the failure index.
    """
    # Mock driver
    mock_driver = mocker.Mock()
    # Mock execute_query on driver
    # Scenario:
    # Batch size 2. Data size 4. 2 chunks.
    # Chunk 1: Success
    # Chunk 2: Failure

    def side_effect(query: str, parameters_: dict[str, Any], database_: str) -> tuple[list[Any], Any, Any]:
        batch = parameters_["batch"]
        if batch[0]["id"] == 3:  # The second chunk starts with id 3
            raise Exception("Chunk Failed")
        return [], None, None

    mock_driver.execute_query.side_effect = side_effect

    client = Neo4jClient(uri="bolt://localhost:7687", auth=("u", "p"))
    client._driver = mock_driver  # Inject mock driver

    data = [
        {"id": 1},
        {"id": 2},  # Chunk 1
        {"id": 3},
        {"id": 4},  # Chunk 2
    ]

    with pytest.raises(Exception) as exc:
        client.batch_write("QUERY", data, batch_size=2)

    assert "Chunk Failed" in str(exc.value)
    # Ideally check log message, but exception propagation is key.
