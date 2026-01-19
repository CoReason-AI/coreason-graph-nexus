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
from unittest.mock import AsyncMock, MagicMock

import pytest

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClientAsync
from coreason_graph_nexus.interfaces import OntologyResolver, SourceAdapter
from coreason_graph_nexus.models import GraphJob, LinkPredictionRequest, ProjectionManifest
from coreason_graph_nexus.service import Service, ServiceAsync

# --- Mocks ---


class MockOntologyResolver(OntologyResolver):
    def resolve(self, term: str) -> tuple[str | None, bool]:
        return "ID_" + term, False


class MockSourceAdapter(SourceAdapter):
    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def read_table(self, table_name: str) -> Any:
        yield {"id": "1", "name": "Test"}


# --- Fixtures ---


@pytest.fixture
def mock_resolver() -> MockOntologyResolver:
    return MockOntologyResolver()


@pytest.fixture
def mock_client_async() -> MagicMock:
    client = MagicMock(spec=Neo4jClientAsync)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock()
    client.close = AsyncMock()
    client.verify_connectivity = AsyncMock()
    client.execute_query = AsyncMock(return_value=[])
    client.batch_write = AsyncMock()
    client.merge_nodes = AsyncMock()
    client.merge_relationships = AsyncMock()
    return client


@pytest.fixture
def service_async(mock_resolver: MockOntologyResolver, mock_client_async: MagicMock) -> ServiceAsync:
    return ServiceAsync(resolver=mock_resolver, client=mock_client_async)


@pytest.fixture
def service_sync(mock_resolver: MockOntologyResolver, mock_client_async: MagicMock) -> Service:
    return Service(resolver=mock_resolver, client=mock_client_async)


# --- Tests ---


@pytest.mark.asyncio
async def test_service_async_lifecycle(service_async: ServiceAsync, mock_client_async: MagicMock) -> None:
    """Test ServiceAsync context manager lifecycle (external client)."""
    async with service_async as svc:
        assert svc is service_async
        # External client -> not managed by service
        mock_client_async.__aenter__.assert_not_called()

    mock_client_async.__aexit__.assert_not_called()


@pytest.mark.asyncio
async def test_service_async_lifecycle_internal(mocker: Any, mock_resolver: MockOntologyResolver) -> None:
    """Test ServiceAsync context manager lifecycle (internal client)."""
    mock_client_cls = mocker.patch("coreason_graph_nexus.service.Neo4jClientAsync")
    mock_instance = mock_client_cls.return_value
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock()

    svc = ServiceAsync(resolver=mock_resolver)  # No client passed -> internal

    async with svc:
        mock_instance.__aenter__.assert_called_once()

    mock_instance.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_service_async_delegation(
    mocker: Any, mock_resolver: MockOntologyResolver, mock_client_async: MagicMock
) -> None:
    """Test that ServiceAsync delegates to Projector/Predictor."""

    # Patch the classes used in __init__
    mock_proj_cls = mocker.patch("coreason_graph_nexus.service.ProjectionEngineAsync")
    mock_proj_instance = mock_proj_cls.return_value
    mock_proj_instance.ingest_entities = AsyncMock()
    mock_proj_instance.ingest_relationships = AsyncMock()

    mock_pred_cls = mocker.patch("coreason_graph_nexus.service.LinkPredictorAsync")
    mock_pred_instance = mock_pred_cls.return_value
    mock_pred_instance.predict_links = AsyncMock()

    svc = ServiceAsync(resolver=mock_resolver, client=mock_client_async)

    manifest = MagicMock(spec=ProjectionManifest)
    adapter = MagicMock(spec=SourceAdapter)
    job = MagicMock(spec=GraphJob)
    job.id = "test-job-id"  # Fix AttributeError

    # Test Ingest Entities
    await svc.ingest_entities(manifest, adapter, job)
    cast(AsyncMock, svc.projector.ingest_entities).assert_called_once_with(manifest, adapter, job, 10000)

    # Test Ingest Relationships
    await svc.ingest_relationships(manifest, adapter, job)
    cast(AsyncMock, svc.projector.ingest_relationships).assert_called_once_with(manifest, adapter, job, 10000)

    # Test Predict Links
    req = MagicMock(spec=LinkPredictionRequest)
    await svc.predict_links(req)
    cast(AsyncMock, svc.predictor.predict_links).assert_called_once_with(req)


def test_service_sync_facade(mocker: Any, mock_resolver: MockOntologyResolver, mock_client_async: MagicMock) -> None:
    """Test Service (Sync) facade delegates via anyio.run."""

    # We need to mock ServiceAsync inside Service.
    mock_svc_async_cls = mocker.patch("coreason_graph_nexus.service.ServiceAsync")
    mock_svc_async = mock_svc_async_cls.return_value
    mock_svc_async.__aenter__ = AsyncMock()
    mock_svc_async.__aexit__ = AsyncMock()
    mock_svc_async.ingest_entities = AsyncMock()
    mock_svc_async.ingest_relationships = AsyncMock()
    mock_svc_async.predict_links = AsyncMock()

    svc = Service(resolver=mock_resolver, client=mock_client_async)

    # Test Context Manager
    with svc:
        pass

    assert mock_svc_async.__aenter__.call_count == 1
    assert mock_svc_async.__aexit__.call_count == 1

    # Test method delegation
    manifest = MagicMock(spec=ProjectionManifest)
    adapter = MagicMock(spec=SourceAdapter)
    job = MagicMock(spec=GraphJob)

    svc.ingest_entities(manifest, adapter, job)
    mock_svc_async.ingest_entities.assert_called_with(manifest, adapter, job, 10000)

    svc.ingest_relationships(manifest, adapter, job)
    mock_svc_async.ingest_relationships.assert_called_with(manifest, adapter, job, 10000)

    req = MagicMock(spec=LinkPredictionRequest)
    svc.predict_links(req)
    mock_svc_async.predict_links.assert_called_with(req)
