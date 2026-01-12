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
from coreason_graph_nexus.interfaces import OntologyResolver, SourceAdapter
from coreason_graph_nexus.models import GraphJob, ProjectionManifest
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

    def ingest_entities(
        self,
        manifest: ProjectionManifest,
        adapter: SourceAdapter,
        job: GraphJob,
        batch_size: int = 10000,
    ) -> None:
        """
        Ingests all entities defined in the manifest.

        Reads from the source adapter, resolves identities, and writes to Neo4j.

        Args:
            manifest: The projection configuration.
            adapter: The source data adapter.
            job: The job tracking object (metrics are updated in-place).
            batch_size: Number of records per transaction.
        """
        logger.info(f"Starting entity ingestion for Job {job.id}")
        job.status = "PROJECTING"

        for entity in manifest.entities:
            logger.info(f"Processing Entity: {entity.name} (Source: {entity.source_table})")

            # Prepare buffers
            batch_data: list[dict[str, Any]] = []

            # Read from source
            try:
                # We need to assume the adapter is connected or managed by the caller
                # But typically the caller manages the context.
                row_iterator = adapter.read_table(entity.source_table)

                for row in row_iterator:
                    # 1. Extract ID
                    source_id = row.get(entity.id_column)
                    if not source_id:
                        logger.warning(f"Skipping row with missing ID in {entity.source_table}")
                        continue

                    # 2. Resolve Ontology (Optional but recommended by PRD)
                    # The PRD says "Before creation, nodes are resolved against the Master Ontology."
                    # We use the resolver. If it returns None, we might fallback to source_id
                    # or skip? For now, let's use resolved ID if available, else source_id.
                    # Ideally, we store both: `id` (canonical) and `source_id`.

                    term_to_resolve = str(source_id)
                    resolved_id = self.resolver.resolve(term_to_resolve)

                    final_id = resolved_id if resolved_id else term_to_resolve

                    if not resolved_id:
                        job.metrics["ontology_misses"] = float(job.metrics["ontology_misses"]) + 1.0
                    else:
                        # PRD mentions "ontology_cache_hits" but model has "ontology_misses"
                        # We track misses as per model.
                        pass

                    # 3. Map Properties
                    node_props = {}
                    # Always include the primary ID
                    node_props["id"] = final_id
                    # Also keep source_id for lineage if needed, but strict mapping:
                    # node_props["_source_id"] = source_id

                    for prop_map in entity.properties:
                        if prop_map.source in row:
                            node_props[prop_map.target] = row[prop_map.source]

                    batch_data.append(node_props)

                    # Flush batch
                    if len(batch_data) >= batch_size:
                        self._flush_nodes(entity.name, batch_data, batch_size)
                        job.metrics["nodes_created"] = float(job.metrics["nodes_created"]) + len(batch_data)
                        batch_data = []

                # Flush remaining
                if batch_data:
                    self._flush_nodes(entity.name, batch_data, batch_size)
                    job.metrics["nodes_created"] = float(job.metrics["nodes_created"]) + len(batch_data)

            except Exception as e:
                logger.error(f"Failed to ingest entity {entity.name}: {e}")
                # We do not stop the whole job? Or do we?
                # PRD doesn't specify failure mode. We raise to be safe.
                raise

        logger.info(f"Entity ingestion complete. Nodes created: {job.metrics['nodes_created']}")

    def _flush_nodes(self, label: str, data: list[dict[str, Any]], batch_size: int) -> None:
        """Helper to write a batch of nodes."""
        # We use merge_nodes with "id" as the merge key
        self.client.merge_nodes(label, data, merge_keys=["id"], batch_size=batch_size)
