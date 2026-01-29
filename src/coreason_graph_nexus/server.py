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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import redis
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from coreason_graph_nexus.adapters.parquet_adapter import ParquetAdapter
from coreason_graph_nexus.config import settings
from coreason_graph_nexus.models import (
    GraphAnalysisRequest,
    GraphJob,
    LinkPredictionRequest,
    ProjectionManifest,
)
from coreason_graph_nexus.ontology import RedisOntologyResolver
from coreason_graph_nexus.service import ServiceAsync
from coreason_graph_nexus.utils.logger import configure_logging, logger


class IngestRequest(BaseModel):
    manifest: ProjectionManifest
    source_base_path: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Configure logging
    configure_logging()
    logger.info("Starting Coreason Graph Nexus Service...")

    # Initialize Redis (Sync client for RedisOntologyResolver)
    try:
        redis_client = redis.Redis.from_url(settings.redis_url)
        redis_client.ping()
        logger.info(f"Connected to Redis at {settings.redis_url}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise RuntimeError("Redis connection failed") from e

    # Initialize Resolver
    resolver = RedisOntologyResolver(redis_client)

    # Initialize ServiceAsync
    service = ServiceAsync(resolver)

    # Initialize Service Resources (Neo4j)
    await service.__aenter__()

    # Store in app state
    app.state.nexus = service
    app.state.redis = redis_client

    yield

    # Cleanup
    logger.info("Shutting down Coreason Graph Nexus Service...")
    await service.__aexit__(None, None, None)
    redis_client.close()


app = FastAPI(title="Coreason Graph Nexus", lifespan=lifespan)


@app.post("/project/ingest", response_model=GraphJob)
async def project_ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> GraphJob:
    service: ServiceAsync = app.state.nexus
    manifest = request.manifest

    # Handle source base path if provided
    # We update the manifest in-memory to prepend base path to source_table
    if request.source_base_path:
        base = Path(request.source_base_path)
        for entity in manifest.entities:
            # We convert to string using as_posix() to ensure consistent path separators (forward slash)
            # This is safer for cross-platform compatibility and testing assertions.
            entity.source_table = (base / entity.source_table).as_posix()
        for rel in manifest.relationships:
            rel.source_table = (base / rel.source_table).as_posix()

    job_id = uuid.uuid4()
    job = GraphJob(id=job_id, manifest_path="api_request", status="PROJECTING")

    async def run_ingest(manifest: ProjectionManifest, job: GraphJob) -> None:
        # Use ParquetAdapter
        adapter = ParquetAdapter()
        try:
            # Ingest Entities
            await service.ingest_entities(manifest, adapter, job)

            # Ingest Relationships
            await service.ingest_relationships(manifest, adapter, job)

            job.status = "COMPLETE"
        except Exception as e:
            logger.error(f"Ingestion failed for job {job.id}: {e}")
            job.status = "FAILED"

    background_tasks.add_task(run_ingest, manifest, job)

    return job


@app.post("/compute/analysis")
async def compute_analysis(request: GraphAnalysisRequest) -> Any:
    service: ServiceAsync = app.state.nexus
    try:
        result = await service.run_analysis(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/links")
async def predict_links(request: LinkPredictionRequest) -> dict[str, str]:
    service: ServiceAsync = app.state.nexus
    try:
        await service.predict_links(request)
        return {"status": "success", "message": "Link prediction complete"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Link prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check() -> dict[str, str]:
    service: ServiceAsync = app.state.nexus
    redis_client: redis.Redis = app.state.redis

    status = {"status": "ok", "neo4j": "unknown", "redis": "unknown"}

    # Check Redis
    try:
        redis_client.ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {e}"
        status["status"] = "degraded"

    # Check Neo4j
    try:
        # ServiceAsync internal client is protected but we can access it via _client or if exposed.
        # ServiceAsync doesn't expose execute_query directly, but we can access _client.
        # Or better, add a health check method to ServiceAsync?
        # For now, accessing _client.
        await service._client.execute_query("MATCH (n) RETURN count(n) LIMIT 1")
        status["neo4j"] = "ok"
    except Exception as e:
        status["neo4j"] = f"error: {e}"
        status["status"] = "degraded"

    if status["status"] != "ok":
        # Return 503 if degraded? Or just 200 with details?
        # Usually 200 is fine if application is reachable, but load balancers might prefer 503.
        # I'll stick to 200 with details for "Deep Health Check" endpoint unless specified.
        pass

    return status
