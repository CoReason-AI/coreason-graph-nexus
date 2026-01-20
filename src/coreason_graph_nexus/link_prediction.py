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

import anyio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient, Neo4jClientAsync
from coreason_graph_nexus.models import LinkPredictionMethod, LinkPredictionRequest
from coreason_graph_nexus.utils.logger import logger


class LinkPredictor:
    """
    The Link Predictor ("The Analyst") - Sync Version.
    """

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    def predict_links(self, request: LinkPredictionRequest) -> None:
        logger.info(f"Starting link prediction using method: {request.method.value}")

        if request.method == LinkPredictionMethod.HEURISTIC:
            if not request.heuristic_query:
                raise ValueError("Heuristic query is missing.")
            self._run_heuristic(request.heuristic_query)
        elif request.method == LinkPredictionMethod.SEMANTIC:
            self._run_semantic(request)
        else:
            raise NotImplementedError(f"Method {request.method} is not implemented.")

    def _run_heuristic(self, query: str) -> None:
        logger.info("Executing heuristic rule...")
        logger.debug(f"Query: {query}")
        try:
            self.client.execute_query(query)
            logger.info("Heuristic rule execution complete.")
        except Exception as e:
            logger.error(f"Failed to execute heuristic rule: {e}")
            raise

    def _run_semantic(self, request: LinkPredictionRequest) -> None:
        if not request.source_label or not request.target_label:
            raise ValueError("source_label and target_label are required.")

        logger.info(
            f"Executing semantic prediction: {request.source_label} <-> {request.target_label} "
            f"(threshold={request.threshold})"
        )

        source_data = self._fetch_embeddings(request.source_label, request.embedding_property)
        if not source_data:
            logger.warning(f"No embeddings found for source label: {request.source_label}")
            return

        target_data = self._fetch_embeddings(request.target_label, request.embedding_property)
        if not target_data:
            logger.warning(f"No embeddings found for target label: {request.target_label}")
            return

        logger.info(
            f"Computing cosine similarity between {len(source_data)} source and {len(target_data)} target nodes."
        )

        relationships_to_create = _compute_similarities(source_data, target_data, request.threshold)

        logger.info(f"Found {len(relationships_to_create)} implicit relationships above threshold.")

        if not relationships_to_create:
            return

        query = (
            f"UNWIND $batch AS row "
            f"MATCH (source), (target) "
            f"WHERE elementId(source) = row.start_id AND elementId(target) = row.end_id "
            f"MERGE (source)-[r:`{request.relationship_type}`]->(target) "
            f"SET r.score = row.score"
        )

        self.client.batch_write(query, relationships_to_create, batch_size=5000)
        logger.info("Semantic link prediction complete.")

    def _fetch_embeddings(self, label: str, property_key: str) -> list[dict[str, Any]]:
        query = (
            f"MATCH (n:`{label}`) "
            f"WHERE n.`{property_key}` IS NOT NULL "
            f"RETURN elementId(n) as id, n.`{property_key}` as embedding"
        )
        return self.client.execute_query(query)


class LinkPredictorAsync:
    """
    The Link Predictor ("The Analyst") - Async Version.
    """

    def __init__(self, client: Neo4jClientAsync) -> None:
        self.client = client

    async def predict_links(self, request: LinkPredictionRequest) -> None:
        logger.info(f"Starting link prediction using method: {request.method.value}")

        if request.method == LinkPredictionMethod.HEURISTIC:
            if not request.heuristic_query:
                raise ValueError("Heuristic query is missing.")
            await self._run_heuristic(request.heuristic_query)
        elif request.method == LinkPredictionMethod.SEMANTIC:
            await self._run_semantic(request)
        else:
            raise NotImplementedError(f"Method {request.method} is not implemented.")

    async def _run_heuristic(self, query: str) -> None:
        logger.info("Executing heuristic rule...")
        logger.debug(f"Query: {query}")
        try:
            await self.client.execute_query(query)
            logger.info("Heuristic rule execution complete.")
        except Exception as e:
            logger.error(f"Failed to execute heuristic rule: {e}")
            raise

    async def _run_semantic(self, request: LinkPredictionRequest) -> None:
        if not request.source_label or not request.target_label:
            raise ValueError("source_label and target_label are required.")

        logger.info(
            f"Executing semantic prediction: {request.source_label} <-> {request.target_label} "
            f"(threshold={request.threshold})"
        )

        source_data = await self._fetch_embeddings(request.source_label, request.embedding_property)
        if not source_data:
            logger.warning(f"No embeddings found for source label: {request.source_label}")
            return

        target_data = await self._fetch_embeddings(request.target_label, request.embedding_property)
        if not target_data:
            logger.warning(f"No embeddings found for target label: {request.target_label}")
            return

        logger.info(
            f"Computing cosine similarity between {len(source_data)} source and {len(target_data)} target nodes."
        )

        # Offload CPU-heavy compute to a worker thread
        relationships_to_create = await anyio.to_thread.run_sync(
            _compute_similarities, source_data, target_data, request.threshold
        )

        logger.info(f"Found {len(relationships_to_create)} implicit relationships above threshold.")

        if not relationships_to_create:
            return

        query = (
            f"UNWIND $batch AS row "
            f"MATCH (source), (target) "
            f"WHERE elementId(source) = row.start_id AND elementId(target) = row.end_id "
            f"MERGE (source)-[r:`{request.relationship_type}`]->(target) "
            f"SET r.score = row.score"
        )

        await self.client.batch_write(query, relationships_to_create, batch_size=5000)
        logger.info("Semantic link prediction complete.")

    async def _fetch_embeddings(self, label: str, property_key: str) -> list[dict[str, Any]]:
        query = (
            f"MATCH (n:`{label}`) "
            f"WHERE n.`{property_key}` IS NOT NULL "
            f"RETURN elementId(n) as id, n.`{property_key}` as embedding"
        )
        return await self.client.execute_query(query)


def _compute_similarities(
    source_data: list[dict[str, Any]], target_data: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """
    Computes cosine similarity and filters results.
    Extracted to be run in a thread/process.
    """
    source_ids = [item["id"] for item in source_data]
    source_vecs = np.array([item["embedding"] for item in source_data])

    target_ids = [item["id"] for item in target_data]
    target_vecs = np.array([item["embedding"] for item in target_data])

    # Result shape: (n_source, n_target)
    similarity_matrix = cosine_similarity(source_vecs, target_vecs)

    rows, cols = np.where(similarity_matrix >= threshold)

    relationships_to_create = []
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
    return relationships_to_create
