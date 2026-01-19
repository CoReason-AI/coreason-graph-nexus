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

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient
from coreason_graph_nexus.config import settings
from coreason_graph_nexus.interfaces import OntologyResolver, SourceAdapter
from coreason_graph_nexus.models import Entity, GraphJob, ProjectionManifest, Relationship
from coreason_graph_nexus.utils.batching import process_and_batch
from coreason_graph_nexus.utils.logger import logger


class ProjectionEngine:
    """
    The Projection Engine ("The Builder").

    Responsible for transforming tabular data from Source Adapters
    into Nodes and Relationships in the Neo4j Graph.
    """

    def __init__(
        self,
        client: Neo4jClient,
        resolver: OntologyResolver,
    ) -> None:
        """
        Initialize the ProjectionEngine.

        Args:
            client: An initialized Neo4jClient.
            resolver: An initialized OntologyResolver.
        """
        self.client = client
        self.resolver = resolver

    async def ingest_entities(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        job: GraphJob,
        batch_size: int = settings.default_batch_size,
    ) -> None:
        """
        Ingests all entities defined in the manifest.

        Reads from the source adapter, resolves identities, and writes to Neo4j.

        Args:
            manifest: The projection configuration.
            adapter: The source data adapter.
            job: The job tracking object (metrics are updated in-place).
            batch_size: Number of records per transaction.

        Raises:
            Exception: If ingestion fails during batch processing.
        """
        logger.info(f"Starting entity ingestion for Job {job.id}")
        job.status = "PROJECTING"

        for entity in manifest.entities:
            logger.info(f"Processing Entity: {entity.name} (Source: {entity.source_table})")

            try:
                row_iterator = adapter.read_table(entity.source_table)

                # Processor Closure
                async def processor(row: dict[str, Any], entity: Entity = entity) -> dict[str, Any] | None:
                    return await self._process_entity_row(row, entity, job)

                # Consumer Closure
                async def consumer(batch: list[dict[str, Any]], entity: Entity = entity) -> None:
                    await self._flush_nodes(entity.name, batch, batch_size)
                    job.metrics["nodes_created"] = float(job.metrics.get("nodes_created", 0.0)) + len(batch)

                await process_and_batch(row_iterator, processor, consumer, batch_size)

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
        """
        Ingests all relationships defined in the manifest.

        Args:
            manifest: The projection configuration.
            adapter: The source data adapter.
            job: The job tracking object (metrics are updated in-place).
            batch_size: Number of records per transaction.

        Raises:
            Exception: If ingestion fails during batch processing.
        """
        logger.info(f"Starting relationship ingestion for Job {job.id}")
        if job.status != "PROJECTING":
            job.status = "PROJECTING"

        for rel in manifest.relationships:
            logger.info(f"Processing Relationship: {rel.name} ({rel.start_node} -> {rel.end_node})")

            try:
                row_iterator = adapter.read_table(rel.source_table)

                # Processor Closure
                async def processor(row: dict[str, Any], rel: Relationship = rel) -> dict[str, Any] | None:
                    return await self._process_relationship_row(row, rel, job)

                # Consumer Closure
                async def consumer(batch: list[dict[str, Any]], rel: Relationship = rel) -> None:
                    await self._flush_relationships(rel, batch, batch_size)
                    job.metrics["edges_created"] = float(job.metrics.get("edges_created", 0.0)) + len(batch)

                await process_and_batch(row_iterator, processor, consumer, batch_size)

            except Exception as e:
                logger.error(f"Failed to ingest relationship {rel.name}: {e}")
                raise

        logger.info(f"Relationship ingestion complete. Edges created: {job.metrics.get('edges_created', 0)}")

    async def _process_entity_row(self, row: dict[str, Any], entity: Entity, job: GraphJob) -> dict[str, Any] | None:
        """
        Processes a single row for entity ingestion.

        Args:
            row: The raw data row.
            entity: The entity configuration.
            job: The graph job object.

        Returns:
            The processed dict or None if row should be skipped.
        """
        source_id = row.get(entity.id_column)
        if source_id is None or source_id == "":
            logger.warning(f"Skipping row with missing ID in {entity.source_table}")
            return None

        # Resolve Ontology
        term_to_resolve = str(source_id)
        resolved_id, is_cache_hit = await self.resolver.resolve(term_to_resolve)

        final_id = resolved_id if resolved_id else term_to_resolve

        if not resolved_id:
            job.metrics["ontology_misses"] = float(job.metrics.get("ontology_misses", 0.0)) + 1.0
        elif is_cache_hit:
            job.metrics["ontology_cache_hits"] = float(job.metrics.get("ontology_cache_hits", 0.0)) + 1.0

        # Map Properties
        node_props = {}
        node_props["id"] = final_id

        for prop_map in entity.properties:
            if prop_map.source in row:
                node_props[prop_map.target] = row[prop_map.source]

        return node_props

    async def _process_relationship_row(
        self, row: dict[str, Any], rel: Relationship, job: GraphJob
    ) -> dict[str, Any] | None:
        """
        Processes a single row for relationship ingestion.

        Args:
            row: The raw data row.
            rel: The relationship configuration.
            job: The graph job object.

        Returns:
            The processed dict or None if row should be skipped.
        """
        source_start = row.get(rel.start_key)
        source_end = row.get(rel.end_key)

        if source_start is None or source_start == "" or source_end is None or source_end == "":
            logger.warning(f"Skipping row with missing keys in {rel.source_table}")
            return None

        # Resolve Start Node
        start_term = str(source_start)
        resolved_start, is_start_cache_hit = await self.resolver.resolve(start_term)
        if not resolved_start:
            job.metrics["ontology_misses"] = float(job.metrics.get("ontology_misses", 0.0)) + 1.0
            final_start = start_term
        else:
            final_start = resolved_start
            if is_start_cache_hit:
                job.metrics["ontology_cache_hits"] = float(job.metrics.get("ontology_cache_hits", 0.0)) + 1.0

        # Resolve End Node
        end_term = str(source_end)
        resolved_end, is_end_cache_hit = await self.resolver.resolve(end_term)
        if not resolved_end:
            job.metrics["ontology_misses"] = float(job.metrics.get("ontology_misses", 0.0)) + 1.0
            final_end = end_term
        else:
            final_end = resolved_end
            if is_end_cache_hit:
                job.metrics["ontology_cache_hits"] = float(job.metrics.get("ontology_cache_hits", 0.0)) + 1.0

        # Prepare Relationship Properties
        rel_props = row.copy()
        rel_props[rel.start_key] = final_start
        rel_props[rel.end_key] = final_end

        return rel_props

    async def _flush_nodes(self, label: str, data: list[dict[str, Any]], batch_size: int) -> None:
        """Helper to write a batch of nodes."""
        await self.client.merge_nodes(label, data, merge_keys=["id"], batch_size=batch_size)

    async def _flush_relationships(self, rel: Relationship, data: list[dict[str, Any]], batch_size: int) -> None:
        """Helper to write a batch of relationships."""
        await self.client.merge_relationships(
            start_label=rel.start_node,
            start_data_key=rel.start_key,
            end_label=rel.end_node,
            end_data_key=rel.end_key,
            rel_type=rel.name,
            data=data,
            batch_size=batch_size,
        )
