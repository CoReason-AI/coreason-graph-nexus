# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from types import TracebackType
from typing import Any, Self

import anyio

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClientAsync
from coreason_graph_nexus.compute import GraphComputerAsync
from coreason_graph_nexus.config import settings
from coreason_graph_nexus.interfaces import OntologyResolver, SourceAdapter
from coreason_graph_nexus.link_prediction import LinkPredictorAsync
from coreason_graph_nexus.models import (
    GraphAnalysisRequest,
    GraphJob,
    LinkPredictionRequest,
    ProjectionManifest,
)
from coreason_graph_nexus.projector import ProjectionEngineAsync


class ServiceAsync:
    """
    The Core Async Service for the Graph Nexus.

    This class orchestrates the Graph Nexus components (Projector, LinkPredictor)
    and manages resources (Neo4j Connection) using the Async Context Manager protocol.
    """

    def __init__(
        self,
        resolver: OntologyResolver,
        client: Neo4jClientAsync | None = None,
    ) -> None:
        """
        Initialize the ServiceAsync.

        Args:
            resolver: The OntologyResolver instance (required).
            client: Optional Neo4jClientAsync. If not provided, one will be created.
        """
        self._internal_client = client is None
        self._client = client or Neo4jClientAsync(
            uri=settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self.projector = ProjectionEngineAsync(self._client, resolver)
        self.predictor = LinkPredictorAsync(self._client)
        self.computer = GraphComputerAsync(self._client)

    async def __aenter__(self) -> Self:
        if self._internal_client:
            await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._internal_client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def ingest_entities(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        job: GraphJob,
        batch_size: int = settings.default_batch_size,
    ) -> None:
        """Async wrapper for entity ingestion."""
        await self.projector.ingest_entities(manifest, adapter, job, batch_size)

    async def ingest_relationships(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        job: GraphJob,
        batch_size: int = settings.default_batch_size,
    ) -> None:
        """Async wrapper for relationship ingestion."""
        await self.projector.ingest_relationships(manifest, adapter, job, batch_size)

    async def predict_links(self, request: LinkPredictionRequest) -> None:
        """Async wrapper for link prediction."""
        await self.predictor.predict_links(request)

    async def run_analysis(self, request: GraphAnalysisRequest) -> Any:
        """Async wrapper for graph analysis."""
        return await self.computer.run_analysis(request)


class Service:
    """
    The Synchronous Facade for the Graph Nexus Service.

    Wraps ServiceAsync and executes methods via `anyio.run`.
    """

    def __init__(
        self,
        resolver: OntologyResolver,
        client: Neo4jClientAsync | None = None,
    ) -> None:
        """
        Initialize the Service.

        Args:
            resolver: The OntologyResolver instance.
            client: Optional Neo4jClientAsync (if sharing an async client, though less common in sync usage).
        """
        self._async_service = ServiceAsync(resolver, client)

    def __enter__(self) -> Self:
        anyio.run(self._async_service.__aenter__)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        anyio.run(self._async_service.__aexit__, exc_type, exc_val, exc_tb)

    def ingest_entities(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        job: GraphJob,
        batch_size: int = settings.default_batch_size,
    ) -> None:
        anyio.run(self._async_service.ingest_entities, manifest, adapter, job, batch_size)

    def ingest_relationships(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        job: GraphJob,
        batch_size: int = settings.default_batch_size,
    ) -> None:
        anyio.run(self._async_service.ingest_relationships, manifest, adapter, job, batch_size)

    def predict_links(self, request: LinkPredictionRequest) -> None:
        anyio.run(self._async_service.predict_links, request)

    def run_analysis(self, request: GraphAnalysisRequest) -> Any:
        return anyio.run(self._async_service.run_analysis, request)
