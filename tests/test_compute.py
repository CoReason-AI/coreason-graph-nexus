# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from typing import Any, cast
from unittest.mock import MagicMock

import networkx as nx
import pytest
from pytest_mock import MockerFixture

from coreason_graph_nexus.compute import GraphComputer
from coreason_graph_nexus.models import AnalysisAlgo, GraphAnalysisRequest


@pytest.fixture
def mock_client(mocker: MockerFixture) -> MagicMock:
    client = mocker.Mock()
    client.to_networkx = mocker.AsyncMock()
    client.batch_write = mocker.AsyncMock()
    return cast(MagicMock, client)


@pytest.mark.asyncio
async def test_pagerank_computation(mock_client: MagicMock) -> None:
    # Setup graph
    G = nx.DiGraph()
    G.add_edge("1", "2")
    G.add_edge("2", "1")

    mock_client.to_networkx.return_value = G

    computer = GraphComputer(mock_client)
    request = GraphAnalysisRequest(
        center_node_id="1",
        algorithm=AnalysisAlgo.PAGERANK,
        depth=1,
    )

    result = await computer.run_analysis(request)

    assert len(result) == 2
    mock_client.batch_write.assert_awaited_once()


@pytest.mark.asyncio
async def test_shortest_path(mock_client: MagicMock) -> None:
    G = nx.DiGraph()
    G.add_node("n1", id="start")
    G.add_node("n2", id="end")
    G.add_edge("n1", "n2")

    mock_client.to_networkx.return_value = G

    computer = GraphComputer(mock_client)
    request = GraphAnalysisRequest(
        center_node_id="start",
        target_node_id="end",
        algorithm=AnalysisAlgo.SHORTEST_PATH,
        depth=1,
    )

    result = await computer.run_analysis(request)
    assert result == ["n1", "n2"]


@pytest.mark.asyncio
async def test_shortest_path_missing_target(mock_client: MagicMock) -> None:
    computer = GraphComputer(mock_client)
    request = GraphAnalysisRequest(
        center_node_id="start",
        algorithm=AnalysisAlgo.SHORTEST_PATH,
        # target_node_id default None
    )

    mock_client.to_networkx.return_value = nx.DiGraph()  # Mock fetch

    with pytest.raises(ValueError, match="target_node_id is required"):
        await computer.run_analysis(request)


@pytest.mark.asyncio
async def test_shortest_path_node_not_found(mock_client: MagicMock) -> None:
    mock_client.to_networkx.return_value = nx.DiGraph()
    computer = GraphComputer(mock_client)
    request = GraphAnalysisRequest(
        center_node_id="start",
        target_node_id="end",
        algorithm=AnalysisAlgo.SHORTEST_PATH,
    )

    with pytest.raises(ValueError, match="Source node 'start' not found"):
        await computer.run_analysis(request)


@pytest.mark.asyncio
async def test_shortest_path_no_path(mock_client: MagicMock, mocker: MockerFixture) -> None:
    G = nx.DiGraph()
    G.add_node("n1", id="start")
    G.add_node("n2", id="end")
    # No edge
    mock_client.to_networkx.return_value = G

    # Patch to_thread.run_sync to run synchronously to ensure coverage tracking works
    # Because pytest-cov might miss coverage inside threads
    async def mock_run_sync(func: Any, *args: Any) -> Any:
        return func(*args)

    mocker.patch("anyio.to_thread.run_sync", side_effect=mock_run_sync)

    computer = GraphComputer(mock_client)
    request = GraphAnalysisRequest(
        center_node_id="start",
        target_node_id="end",
        algorithm=AnalysisAlgo.SHORTEST_PATH,
    )

    result = await computer.run_analysis(request)
    assert result == []


@pytest.mark.asyncio
async def test_louvain(mock_client: MagicMock) -> None:
    G = nx.DiGraph()
    G.add_edge("1", "2")

    mock_client.to_networkx.return_value = G

    computer = GraphComputer(mock_client)
    request = GraphAnalysisRequest(
        center_node_id="1",
        algorithm=AnalysisAlgo.LOUVAIN,
        depth=1,
    )

    result = await computer.run_analysis(request)
    assert len(result) > 0
    mock_client.batch_write.assert_awaited_once()


@pytest.mark.asyncio
async def test_louvain_empty(mock_client: MagicMock) -> None:
    mock_client.to_networkx.return_value = nx.DiGraph()
    computer = GraphComputer(mock_client)
    request = GraphAnalysisRequest(
        center_node_id="1",
        algorithm=AnalysisAlgo.LOUVAIN,
    )
    result = await computer.run_analysis(request)
    assert result == {}


@pytest.mark.asyncio
async def test_not_implemented_algorithm(mock_client: MagicMock) -> None:
    computer = GraphComputer(mock_client)
    request = MagicMock()
    request.algorithm.value = "INVALID"
    request.center_node_id = "1"
    request.depth = 1

    mock_client.to_networkx.return_value = nx.DiGraph()

    with pytest.raises(NotImplementedError):
        await computer.run_analysis(request)
