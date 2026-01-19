# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

"""
orchestration engine for knowledge graph
"""

__version__ = "0.1.0"
__author__ = "Gowtham A Rao"
__email__ = "gowtham.rao@coreason.ai"

from types import TracebackType
from typing import Any, Self, cast

import anyio.from_thread
import httpx

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient
from coreason_graph_nexus.compute import GraphComputer
from coreason_graph_nexus.config import settings
from coreason_graph_nexus.interfaces import OntologyResolver, SourceAdapter
from coreason_graph_nexus.link_prediction import LinkPredictor
from coreason_graph_nexus.main import hello_world
from coreason_graph_nexus.models import (
    GraphAnalysisRequest,
    GraphJob,
    LinkPredictionRequest,
    ProjectionManifest,
)
from coreason_graph_nexus.projector import ProjectionEngine

__all__ = ["hello_world", "Service", "ServiceAsync"]


class ServiceAsync:
    """
    The Core Async Service for CoReason Graph Nexus.

    This service orchestrates the Graph Nexus components (Projector, Link Predictor, Computer)
    in an asyncio-native manner.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """
        Initialize the ServiceAsync.

        Args:
            client: Optional external httpx client for reuse.
        """
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()

        # Initialize Neo4j Client
        self.neo4j = Neo4jClient(
            uri=settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            database=settings.neo4j_database,
        )

        # Components are initialized lazily or on demand if they need specific dependencies
        # But for now we can instantiate them with the neo4j client.

        self.link_predictor = LinkPredictor(self.neo4j)
        self.computer = GraphComputer(self.neo4j)

    async def __aenter__(self) -> Self:
        await self.neo4j.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._internal_client:
            await self._client.aclose()
        await self.neo4j.__aexit__(exc_type, exc_val, exc_tb)

    async def run_projection(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        resolver: OntologyResolver,
        job_id: str,
    ) -> GraphJob:
        """
        Runs a projection job.

        Args:
            manifest: The projection manifest.
            adapter: The source adapter.
            resolver: The ontology resolver.
            job_id: The job ID.

        Returns:
            The completed GraphJob.
        """
        import uuid

        job = GraphJob(id=uuid.UUID(job_id), manifest_path="memory", status="RESOLVING")

        projector = ProjectionEngine(self.neo4j, resolver)

        await projector.ingest_entities(manifest, adapter, job)
        await projector.ingest_relationships(manifest, adapter, job)

        job.status = "COMPLETE"
        return job

    async def run_link_prediction(self, request: LinkPredictionRequest) -> None:
        """
        Runs link prediction.
        """
        await self.link_predictor.predict_links(request)

    async def run_analysis(self, request: GraphAnalysisRequest) -> Any:
        """
        Runs graph analysis.
        """
        return await self.computer.run_analysis(request)


class Service:
    """
    The Synchronous Facade for ServiceAsync.
    """

    def __init__(self, client: Any = None) -> None:
        self._async = ServiceAsync()

    def __enter__(self) -> Self:
        self._portal_cm = anyio.from_thread.start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        self._portal.call(self._async.__aenter__)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if hasattr(self, "_portal"):
            self._portal.call(self._async.__aexit__, exc_type, exc_val, exc_tb)
            self._portal_cm.__exit__(exc_type, exc_val, exc_tb)

    def run_projection(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        resolver: OntologyResolver,
        job_id: str,
    ) -> GraphJob:
        result = self._portal.call(self._async.run_projection, manifest, adapter, resolver, job_id)
        return cast(GraphJob, result)

    def run_link_prediction(self, request: LinkPredictionRequest) -> None:
        self._portal.call(self._async.run_link_prediction, request)

    def run_analysis(self, request: GraphAnalysisRequest) -> Any:
        return self._portal.call(self._async.run_analysis, request)
