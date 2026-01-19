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
from typing import Self

import anyio

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClientAsync
from coreason_graph_nexus.config import settings
from coreason_graph_nexus.interfaces import OntologyResolver, SourceAdapter
from coreason_graph_nexus.link_prediction import LinkPredictorAsync
from coreason_graph_nexus.models import GraphJob, LinkPredictionRequest, ProjectionManifest
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

    async def __aenter__(self) -> Self:
        if self._internal_client:
            await self._client.__aenter__()
        # If external client, we assume it's already managed or will be managed externally,
        # but typical pattern with 'internal_client' flag implies we only manage if we created it.
        # However, if we didn't create it, we might still want to verify connectivity?
        # The Neo4jClientAsync.__aenter__ does verification.
        # If user passed an open client, calling __aenter__ again is usually safe (returns self).
        # But let's follow the prompt pattern: "accept an optional external client/session to allow connection pooling."
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
        # We need to run __aenter__ in the event loop.
        # But anyio.run runs a function and closes the loop (unless specific backend logic).
        # Standard pattern for wrapping async context manager in sync:
        # We can't keep the loop open easily across method calls if we use anyio.run for each call.
        # BUT the prompt says:
        # def __enter__(self):
        #     # Start the event loop for the context ??
        #     return self
        #
        # def __exit__(self, *args):
        #     anyio.run(self._async.__aexit__, *args)
        #
        # def validate_file(self, path):
        #     return anyio.run(self._async.validate_file, path)
        #
        # If I use `anyio.run` in `__enter__`, it will block until `__aenter__` returns.
        # `__aenter__` returns `self`.
        # But `ServiceAsync.__aenter__` sets up the client.
        #
        # The issue with `anyio.run` is that it starts a NEW event loop (or uses existing).
        # If `ServiceAsync` holds resources bound to a loop (like `aiohttp.ClientSession`),
        # calling `anyio.run` multiple times might be problematic if they use different loops.
        # Neo4j driver manages its own connection pool. `AsyncGraphDatabase.driver` creates a driver object.
        # Connectivity verification happens in `__aenter__`.
        #
        # If we use `anyio.run(self._async_service.__aenter__)`, it runs, verifies, and returns.
        # Then `ingest_entities` calls `anyio.run(...)`.
        # If `anyio.run` starts a new loop each time, we need to be careful about resources attached to the loop.
        # `neo4j` async driver: "The driver object is thread-safe and can be shared across multiple threads."
        # But is it loop-bound? Usually async drivers are bound to the loop where they were created or first used.
        # `Neo4jClientAsync` creates driver in `__init__`.
        #
        # If `ServiceAsync` creates driver in `__init__` (sync), it's fine.
        # `__aenter__` calls `verify_connectivity`.
        #
        # If `anyio.run` creates a new loop each time, and the driver is used across loops...
        # `neo4j` async driver might not like switching loops if it holds connections.
        #
        # However, the user instruction is explicit:
        # "Implement __enter__ and __exit__ by running the Async context methods via anyio.run."
        # And "Methods... return anyio.run(self._async.method, ...)"
        #
        # This implies the user accepts this pattern.
        # For simple scripts, `anyio.run` might reuse the loop if `trio` or `asyncio` is used properly,
        # but `anyio.run` typically runs a self-contained entry point.
        #
        # Wait, if `anyio.run` is used for EVERY method call, it means we are starting/stopping loops constantly.
        # This is inefficient but functional if resources aren't loop-tied.
        # `httpx.AsyncClient` IS loop-tied. `neo4j` driver might be.
        #
        # If `Neo4jClientAsync` creates `AsyncGraphDatabase.driver` in `__init__`,
        # it might be fine if the driver isn't used until `execute_query`.
        #
        # Let's check `anyio` docs or knowledge.
        # Usually, you use a background thread with a loop for the "Sync Facade" if you want persistent state.
        # But the prompt specifically asked for `anyio.run` in methods.
        #
        # "Refactor the package... to be Async-First... Service (The Facade): Wraps the Core...
        # Implements __enter__ and __exit__ by running the Async context methods via anyio.run."
        #
        # I will follow instructions.
        # If `neo4j` driver complains about loops, it's a known issue with this pattern,
        # but I must follow the pattern.
        #
        # Actually, `anyio.run` executes the function.
        # `__enter__` cannot easily use `anyio.run` because `__enter__` must return `self`
        # and not block for the whole duration.
        # `__enter__` is sync.
        # The prompt says:
        # def __enter__(self):
        #      # Start the event loop for the context
        #      return self
        #
        # This comment "Start the event loop for the context" suggests maybe starting a thread?
        # But the code example:
        # def __exit__(self, *args):
        #      anyio.run(self._async.__aexit__, *args)
        #
        # This implies `__enter__` might just be `anyio.run(self._async.__aenter__)`.
        # But `__aenter__` returns `self` and exits. The loop closes?
        # If the loop closes, async resources might be closed.
        #
        # However, the user provided code snippet for `validate_file` uses `anyio.run`.
        #
        # I will implement it as requested.
        #
        # `__enter__` calls `anyio.run(self._async_service.__aenter__)`.
        # `__exit__` calls `anyio.run(self._async_service.__aexit__, ...)`
        #
        # Note: `__aenter__` return value is `self._async_service`. We ignore it or check it.
        # `Service` returns `self` (the facade).

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
