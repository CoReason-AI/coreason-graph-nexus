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

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient, Neo4jClientAsync
from coreason_graph_nexus.config import settings
from coreason_graph_nexus.interfaces import OntologyResolver, SourceAdapter
from coreason_graph_nexus.models import Entity, GraphJob, ProjectionManifest, Relationship
from coreason_graph_nexus.utils.batching import process_and_batch, process_and_batch_async
from coreason_graph_nexus.utils.logger import logger


class ProjectionEngine:
    """
    The Projection Engine ("The Builder") - Sync Version.
    """

    def __init__(
        self,
        client: Neo4jClient,
        resolver: OntologyResolver,
    ) -> None:
        self.client = client
        self.resolver = resolver

    def ingest_entities(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        job: GraphJob,
        batch_size: int = settings.default_batch_size,
    ) -> None:
        logger.info(f"Starting entity ingestion for Job {job.id}")
        job.status = "PROJECTING"

        for entity in manifest.entities:
            logger.info(f"Processing Entity: {entity.name} (Source: {entity.source_table})")
            try:
                row_iterator = adapter.read_table(entity.source_table)

                def processor(row: dict[str, Any], entity: Entity = entity) -> dict[str, Any] | None:
                    return _process_entity_row(row, entity, job, self.resolver)

                def consumer(batch: list[dict[str, Any]], entity: Entity = entity) -> None:
                    self.client.merge_nodes(entity.name, batch, merge_keys=["id"], batch_size=batch_size)
                    job.metrics["nodes_created"] = float(job.metrics.get("nodes_created", 0.0)) + len(batch)

                process_and_batch(row_iterator, processor, consumer, batch_size)
            except Exception as e:
                logger.error(f"Failed to ingest entity {entity.name}: {e}")
                raise

        logger.info(f"Entity ingestion complete. Nodes created: {job.metrics.get('nodes_created', 0)}")

    def ingest_relationships(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        job: GraphJob,
        batch_size: int = settings.default_batch_size,
    ) -> None:
        logger.info(f"Starting relationship ingestion for Job {job.id}")
        if job.status != "PROJECTING":
            job.status = "PROJECTING"

        for rel in manifest.relationships:
            logger.info(f"Processing Relationship: {rel.name} ({rel.start_node} -> {rel.end_node})")
            try:
                row_iterator = adapter.read_table(rel.source_table)

                def processor(row: dict[str, Any], rel: Relationship = rel) -> dict[str, Any] | None:
                    return _process_relationship_row(row, rel, job, self.resolver)

                def consumer(batch: list[dict[str, Any]], rel: Relationship = rel) -> None:
                    self.client.merge_relationships(
                        start_label=rel.start_node,
                        start_data_key=rel.start_key,
                        end_label=rel.end_node,
                        end_data_key=rel.end_key,
                        rel_type=rel.name,
                        data=batch,
                        batch_size=batch_size,
                    )
                    job.metrics["edges_created"] = float(job.metrics.get("edges_created", 0.0)) + len(batch)

                process_and_batch(row_iterator, processor, consumer, batch_size)
            except Exception as e:
                logger.error(f"Failed to ingest relationship {rel.name}: {e}")
                raise

        logger.info(f"Relationship ingestion complete. Edges created: {job.metrics.get('edges_created', 0)}")


class ProjectionEngineAsync:
    """
    The Projection Engine ("The Builder") - Async Version.
    """

    def __init__(
        self,
        client: Neo4jClientAsync,
        resolver: OntologyResolver,
    ) -> None:
        self.client = client
        self.resolver = resolver

    async def ingest_entities(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        job: GraphJob,
        batch_size: int = settings.default_batch_size,
    ) -> None:
        logger.info(f"Starting entity ingestion for Job {job.id}")
        job.status = "PROJECTING"

        for entity in manifest.entities:
            logger.info(f"Processing Entity: {entity.name} (Source: {entity.source_table})")
            try:
                # We assume read_table returns a sync iterator.
                row_iterator = adapter.read_table(entity.source_table)

                # Processor (Sync)
                def processor(row: dict[str, Any], entity: Entity = entity) -> dict[str, Any] | None:
                    return _process_entity_row(row, entity, job, self.resolver)

                # Consumer (Async)
                async def consumer(batch: list[dict[str, Any]], entity: Entity = entity) -> None:
                    await self.client.merge_nodes(entity.name, batch, merge_keys=["id"], batch_size=batch_size)
                    job.metrics["nodes_created"] = float(job.metrics.get("nodes_created", 0.0)) + len(batch)

                await process_and_batch_async(row_iterator, processor, consumer, batch_size)
            except Exception as e:
                logger.error(f"Failed to ingest entity {entity.name}: {e}")
                raise

        logger.info(f"Entity ingestion complete. Nodes created: {job.metrics.get('nodes_created', 0)}")

    async def ingest_relationships(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        job: GraphJob,
        batch_size: int = settings.default_batch_size,
    ) -> None:
        logger.info(f"Starting relationship ingestion for Job {job.id}")
        if job.status != "PROJECTING":
            job.status = "PROJECTING"

        for rel in manifest.relationships:
            logger.info(f"Processing Relationship: {rel.name} ({rel.start_node} -> {rel.end_node})")
            try:
                row_iterator = adapter.read_table(rel.source_table)

                # Processor (Sync)
                def processor(row: dict[str, Any], rel: Relationship = rel) -> dict[str, Any] | None:
                    return _process_relationship_row(row, rel, job, self.resolver)

                # Consumer (Async)
                async def consumer(batch: list[dict[str, Any]], rel: Relationship = rel) -> None:
                    await self.client.merge_relationships(
                        start_label=rel.start_node,
                        start_data_key=rel.start_key,
                        end_label=rel.end_node,
                        end_data_key=rel.end_key,
                        rel_type=rel.name,
                        data=batch,
                        batch_size=batch_size,
                    )
                    job.metrics["edges_created"] = float(job.metrics.get("edges_created", 0.0)) + len(batch)

                await process_and_batch_async(row_iterator, processor, consumer, batch_size)
            except Exception as e:
                logger.error(f"Failed to ingest relationship {rel.name}: {e}")
                raise

        logger.info(f"Relationship ingestion complete. Edges created: {job.metrics.get('edges_created', 0)}")


# Shared processing logic (Stateless functions)


def _process_entity_row(
    row: dict[str, Any],
    entity: Entity,
    job: GraphJob,
    resolver: OntologyResolver,
) -> dict[str, Any] | None:
    source_id = row.get(entity.id_column)
    if source_id is None or source_id == "":
        logger.warning(f"Skipping row with missing ID in {entity.source_table}")
        return None

    term_to_resolve = str(source_id)
    resolved_id, is_cache_hit = resolver.resolve(term_to_resolve)

    final_id = resolved_id if resolved_id else term_to_resolve

    if not resolved_id:
        job.metrics["ontology_misses"] = float(job.metrics.get("ontology_misses", 0.0)) + 1.0
    elif is_cache_hit:
        job.metrics["ontology_cache_hits"] = float(job.metrics.get("ontology_cache_hits", 0.0)) + 1.0

    node_props = {}
    node_props["id"] = final_id

    for prop_map in entity.properties:
        if prop_map.source in row:
            node_props[prop_map.target] = row[prop_map.source]

    return node_props


def _process_relationship_row(
    row: dict[str, Any],
    rel: Relationship,
    job: GraphJob,
    resolver: OntologyResolver,
) -> dict[str, Any] | None:
    source_start = row.get(rel.start_key)
    source_end = row.get(rel.end_key)

    if source_start is None or source_start == "" or source_end is None or source_end == "":
        logger.warning(f"Skipping row with missing keys in {rel.source_table}")
        return None

    # Resolve Start Node
    start_term = str(source_start)
    resolved_start, is_start_cache_hit = resolver.resolve(start_term)
    if not resolved_start:
        job.metrics["ontology_misses"] = float(job.metrics.get("ontology_misses", 0.0)) + 1.0
        final_start = start_term
    else:
        final_start = resolved_start
        if is_start_cache_hit:
            job.metrics["ontology_cache_hits"] = float(job.metrics.get("ontology_cache_hits", 0.0)) + 1.0

    # Resolve End Node
    end_term = str(source_end)
    resolved_end, is_end_cache_hit = resolver.resolve(end_term)
    if not resolved_end:
        job.metrics["ontology_misses"] = float(job.metrics.get("ontology_misses", 0.0)) + 1.0
        final_end = end_term
    else:
        final_end = resolved_end
        if is_end_cache_hit:
            job.metrics["ontology_cache_hits"] = float(job.metrics.get("ontology_cache_hits", 0.0)) + 1.0

    rel_props = row.copy()
    rel_props[rel.start_key] = final_start
    rel_props[rel.end_key] = final_end

    return rel_props
