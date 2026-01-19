# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

import uuid
from typing import Any, cast
from unittest.mock import MagicMock

import networkx as nx
import pytest
from pytest_mock import MockerFixture

from coreason_graph_nexus import Service
from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient
from coreason_graph_nexus.compute import GraphComputer
from coreason_graph_nexus.link_prediction import LinkPredictor
from coreason_graph_nexus.models import (
    AnalysisAlgo,
    Entity,
    GraphAnalysisRequest,
    GraphJob,
    LinkPredictionMethod,
    LinkPredictionRequest,
    ProjectionManifest,
    Relationship,
)
from coreason_graph_nexus.projector import ProjectionEngine
from tests.test_utils_shared import MockOntologyResolver, MockSourceAdapter


@pytest.fixture
def mock_client(mocker: MockerFixture) -> MagicMock:
    client = mocker.Mock()
    client.execute_query = mocker.AsyncMock()
    client.batch_write = mocker.AsyncMock()
    client.to_networkx = mocker.AsyncMock()
    return cast(MagicMock, client)


@pytest.fixture
def mock_driver(mocker: MockerFixture) -> Any:
    driver = mocker.AsyncMock()
    mocker.patch("neo4j.AsyncGraphDatabase.driver", return_value=driver)
    driver.execute_query.return_value = ([], None, None)
    return driver


def test_service_facade_all_methods(mock_driver: Any) -> None:
    adapter = MockSourceAdapter({})
    adapter.connected = True
    resolver = MockOntologyResolver({})
    manifest = ProjectionManifest(version="1", source_connection="x", entities=[], relationships=[])

    with Service() as svc:
        svc.run_projection(manifest, adapter, resolver, str(uuid.uuid4()))

        req_lp = LinkPredictionRequest(method=LinkPredictionMethod.HEURISTIC, heuristic_query="Q")
        svc.run_link_prediction(req_lp)

        req_ga = GraphAnalysisRequest(center_node_id="1", algorithm=AnalysisAlgo.PAGERANK)
        # Mocking to_networkx inside ServiceAsync is hard from here without patching ServiceAsync logic
        # But we mock driver.execute_query to return empty.
        # to_networkx calls execute_query.
        # So it should be fine.
        svc.run_analysis(req_ga)


@pytest.mark.asyncio
async def test_heuristic_exception(mock_client: MagicMock) -> None:
    predictor = LinkPredictor(mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.HEURISTIC,
        heuristic_query="Q",
    )
    mock_client.execute_query.side_effect = Exception("Fail")

    with pytest.raises(Exception, match="Fail"):
        await predictor.predict_links(request)


@pytest.mark.asyncio
async def test_semantic_missing_labels(mock_client: MagicMock) -> None:
    predictor = LinkPredictor(mock_client)
    request = LinkPredictionRequest.model_construct(
        method=LinkPredictionMethod.SEMANTIC,
        source_label=None,
        target_label="Target",
    )

    with pytest.raises(ValueError, match="source_label and target_label are required"):
        await predictor.predict_links(request)


@pytest.mark.asyncio
async def test_semantic_target_empty(mock_client: MagicMock) -> None:
    predictor = LinkPredictor(mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC,
        source_label="S",
        target_label="T",
    )
    mock_client.execute_query.side_effect = [
        [{"id": "1", "embedding": [1.0]}],
        [],
    ]
    await predictor.predict_links(request)
    assert mock_client.execute_query.await_count == 2
    mock_client.batch_write.assert_not_awaited()


@pytest.mark.asyncio
async def test_semantic_low_similarity(mock_client: MagicMock) -> None:
    predictor = LinkPredictor(mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC,
        source_label="S",
        target_label="T",
        threshold=0.9,
    )
    mock_client.execute_query.side_effect = [
        [{"id": "1", "embedding": [1.0, 0.0]}],
        [{"id": "2", "embedding": [0.0, 1.0]}],
    ]
    await predictor.predict_links(request)
    mock_client.batch_write.assert_not_awaited()


@pytest.mark.asyncio
async def test_semantic_self_loop(mock_client: MagicMock) -> None:
    predictor = LinkPredictor(mock_client)
    request = LinkPredictionRequest(
        method=LinkPredictionMethod.SEMANTIC,
        source_label="S",
        target_label="T",
        threshold=0.0,
    )
    mock_client.execute_query.side_effect = [
        [{"id": "1", "embedding": [1.0]}],
        [{"id": "1", "embedding": [1.0]}],
    ]
    await predictor.predict_links(request)
    mock_client.batch_write.assert_not_awaited()


@pytest.mark.asyncio
async def test_nx_no_path(mock_client: MagicMock, mocker: MockerFixture) -> None:
    G = nx.DiGraph()
    G.add_node("1", id="start")
    G.add_node("2", id="end")
    mock_client.to_networkx.return_value = G

    computer = GraphComputer(mock_client)
    request = GraphAnalysisRequest(
        center_node_id="start",
        target_node_id="end",
        algorithm=AnalysisAlgo.SHORTEST_PATH,
    )

    async def mock_run_sync(func: Any, *args: Any) -> Any:
        return func(*args)

    mocker.patch("anyio.to_thread.run_sync", side_effect=mock_run_sync)

    res = await computer.run_analysis(request)
    assert res == []


@pytest.mark.asyncio
async def test_compute_missing_target_explicit(mocker: MockerFixture) -> None:
    client = mocker.Mock()
    computer = GraphComputer(client)
    request = GraphAnalysisRequest(
        center_node_id="1",
        algorithm=AnalysisAlgo.SHORTEST_PATH,
    )
    client.to_networkx = mocker.AsyncMock(return_value=nx.DiGraph())

    with pytest.raises(ValueError):
        await computer.run_analysis(request)


@pytest.mark.asyncio
async def test_projector_relationship_missing_keys_coverage(mocker: MockerFixture) -> None:
    adapter = MockSourceAdapter({"knows": [{"p1": None, "p2": "2"}, {"p1": "1", "p2": None}]})
    resolver = MockOntologyResolver({})
    client = mocker.Mock()
    client.merge_relationships = mocker.AsyncMock()

    engine = ProjectionEngine(client, resolver)
    manifest = ProjectionManifest(
        version="1",
        source_connection="x",
        entities=[Entity(name="P", source_table="x", id_column="id", ontology_mapping="none", properties=[])],
        relationships=[
            Relationship(name="R", source_table="knows", start_node="P", start_key="p1", end_node="P", end_key="p2")
        ],
    )
    job = GraphJob(id=uuid.uuid4(), manifest_path="x", status="RESOLVING")

    async with adapter:
        await engine.ingest_relationships(manifest, adapter, job)

    assert job.metrics["edges_created"] == 0.0


@pytest.mark.asyncio
async def test_projector_ontology_miss_coverage(mocker: MockerFixture) -> None:
    adapter = MockSourceAdapter({"knows": [{"p1": "unknown", "p2": "2"}]})
    resolver = mocker.Mock()
    resolver.resolve = mocker.AsyncMock(return_value=(None, False))

    client = mocker.Mock()
    client.merge_relationships = mocker.AsyncMock()

    engine = ProjectionEngine(client, resolver)
    manifest = ProjectionManifest(
        version="1",
        source_connection="x",
        entities=[Entity(name="P", source_table="x", id_column="id", ontology_mapping="none", properties=[])],
        relationships=[
            Relationship(name="R", source_table="knows", start_node="P", start_key="p1", end_node="P", end_key="p2")
        ],
    )
    job = GraphJob(id=uuid.uuid4(), manifest_path="x", status="RESOLVING")

    async with adapter:
        await engine.ingest_relationships(manifest, adapter, job)

    assert job.metrics["ontology_misses"] >= 1.0
    assert job.metrics["edges_created"] == 1.0


@pytest.mark.asyncio
async def test_neo4j_batch_write_exception_handler(mocker: MockerFixture) -> None:
    mock_logger = mocker.patch("coreason_graph_nexus.adapters.neo4j_adapter.logger")
    driver = mocker.AsyncMock()
    driver.execute_query.side_effect = Exception("Boom")
    mocker.patch("neo4j.AsyncGraphDatabase.driver", return_value=driver)

    client = Neo4jClient("x", ("u", "p"))

    with pytest.raises(Exception, match="Boom"):
        await client.batch_write("Q", [{"a": 1}])

    mock_logger.error.assert_called()
