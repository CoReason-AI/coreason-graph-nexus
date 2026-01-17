# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient
from coreason_graph_nexus.models import LinkPredictionMethod, LinkPredictionRequest
from coreason_graph_nexus.utils.logger import logger


class LinkPredictor:
    """
    The Link Predictor ("The Analyst").

    Responsible for inferring implicit edges in the graph using:
    1. Heuristic Rules (Cypher-based pattern matching).
    2. Semantic Similarity (Vector Embeddings - to be implemented).
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
            # Placeholder for future implementation
            raise NotImplementedError("Semantic link prediction is not yet implemented.")
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
