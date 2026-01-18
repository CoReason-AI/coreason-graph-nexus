# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient
from coreason_graph_nexus.models import LinkPredictionMethod, LinkPredictionRequest
from coreason_graph_nexus.utils.logger import logger


class LinkPredictor:
    """
    The Link Predictor ("The Analyst").

    Responsible for inferring implicit edges in the graph using:
    1. Heuristic Rules (Cypher-based pattern matching).
    2. Semantic Similarity (Vector Embeddings).
    """

    def __init__(self, client: Neo4jClient) -> None:
        """
        Initialize the LinkPredictor.

        Args:
            client: An initialized Neo4jClient.
        """
        self.client = client

    def predict_links(self, request: LinkPredictionRequest) -> None:
        """
        Executes the link prediction based on the request configuration.

        Args:
            request: The link prediction request containing method and parameters.
        """
        logger.info(f"Starting link prediction using method: {request.method.value}")

        if request.method == LinkPredictionMethod.HEURISTIC:
            if not request.heuristic_query:
                # Should be caught by validation, but defensive programming
                raise ValueError("Heuristic query is missing.")
            self._run_heuristic(request.heuristic_query)
        elif request.method == LinkPredictionMethod.SEMANTIC:
            self._run_semantic(request)
        else:
            raise NotImplementedError(f"Method {request.method} is not implemented.")

    def _run_heuristic(self, query: str) -> None:
        """
        Executes a Cypher-based heuristic rule.

        Args:
            query: The Cypher query (typically MATCH ... MERGE ...).
        """
        logger.info("Executing heuristic rule...")
        logger.debug(f"Query: {query}")

        try:
            # Execute the query.
            # We assume the query handles the logic (MERGE/CREATE).
            # Neo4jClient.execute_query returns data, but we don't necessarily expect data returned.
            self.client.execute_query(query)
            logger.info("Heuristic rule execution complete.")
        except Exception as e:
            logger.error(f"Failed to execute heuristic rule: {e}")
            raise

    def _run_semantic(self, request: LinkPredictionRequest) -> None:
        """
        Executes semantic link prediction using vector embeddings.

        Args:
            request: The link prediction request.
        """
        if not request.source_label or not request.target_label:
            # Should be caught by validation
            raise ValueError("source_label and target_label are required.")

        logger.info(
            f"Executing semantic prediction: {request.source_label} <-> {request.target_label} "
            f"(threshold={request.threshold})"
        )

        # 1. Fetch Source Embeddings
        source_data = self._fetch_embeddings(request.source_label, request.embedding_property)
        if not source_data:
            logger.warning(f"No embeddings found for source label: {request.source_label}")
            return

        # 2. Fetch Target Embeddings
        target_data = self._fetch_embeddings(request.target_label, request.embedding_property)
        if not target_data:
            logger.warning(f"No embeddings found for target label: {request.target_label}")
            return

        # 3. Compute Similarity
        logger.info(
            f"Computing cosine similarity between {len(source_data)} source and {len(target_data)} target nodes."
        )
        source_ids = [item["id"] for item in source_data]
        source_vecs = np.array([item["embedding"] for item in source_data])

        target_ids = [item["id"] for item in target_data]
        target_vecs = np.array([item["embedding"] for item in target_data])

        # Result shape: (n_source, n_target)
        similarity_matrix = cosine_similarity(source_vecs, target_vecs)

        # 4. Filter and Prepare Write
        relationships_to_create = []

        # Optimize iteration?
        # We can use numpy where to find indices
        # threshold filtering
        rows, cols = np.where(similarity_matrix >= request.threshold)

        for r, c in zip(rows, cols, strict=True):
            score = float(similarity_matrix[r, c])
            s_id = source_ids[r]
            t_id = target_ids[c]

            # Skip self-loops if source and target are the same node
            if s_id == t_id:
                continue

            relationships_to_create.append(
                {
                    "start_id": s_id,
                    "end_id": t_id,
                    "score": score,
                }
            )

        logger.info(f"Found {len(relationships_to_create)} implicit relationships above threshold.")

        if not relationships_to_create:
            return

        # 5. Write to Neo4j
        # We match nodes by their elementId (internal ID) or id property?
        # The _fetch_embeddings returns elementId as 'id'.
        # So we match by elementId(n) = row.start_id

        query = (
            f"UNWIND $batch AS row "
            f"MATCH (source), (target) "
            f"WHERE elementId(source) = row.start_id AND elementId(target) = row.end_id "
            f"MERGE (source)-[r:`{request.relationship_type}`]->(target) "
            f"SET r.score = row.score"
        )

        self.client.batch_write(query, relationships_to_create, batch_size=5000)
        logger.info("Semantic link prediction complete.")

    def _fetch_embeddings(self, label: str, property_key: str) -> list[dict[str, str | list[float]]]:
        """
        Fetches node IDs and their embeddings from Neo4j.

        Args:
            label: The node label.
            property_key: The property containing the embedding.

        Returns:
            List of dicts: {'id': element_id, 'embedding': [...]}
        """
        # We wrap label and property in backticks to be safe
        query = (
            f"MATCH (n:`{label}`) "
            f"WHERE n.`{property_key}` IS NOT NULL "
            f"RETURN elementId(n) as id, n.`{property_key}` as embedding"
        )
        return self.client.execute_query(query)
