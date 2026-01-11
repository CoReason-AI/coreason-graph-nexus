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

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from coreason_graph_nexus.utils.logger import logger


class Neo4jClient:
    """
    A client wrapper for the Neo4j Graph Database.

    This client manages the driver lifecycle and provides a simplified interface
    for executing Cypher queries using best practices (e.g., connection pooling).
    It implements the Context Manager protocol.
    """

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str],
        database: str = "neo4j",
    ) -> None:
        """
        Initialize the Neo4jClient.

        Args:
            uri: The URI of the Neo4j database (e.g., "bolt://localhost:7687").
            auth: A tuple of (username, password).
            database: The name of the database to connect to (default: "neo4j").
        """
        self._uri = uri
        self._auth = auth
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=auth)
        logger.info(f"Initialized Neo4j driver for {uri} (db: {database})")

    def __enter__(self) -> Self:
        """
        Enters the runtime context related to this object.

        Verifies connectivity on entry.
        """
        self.verify_connectivity()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Exits the runtime context and closes the driver.
        """
        self.close()

    def close(self) -> None:
        """
        Closes the underlying Neo4j driver.
        """
        if self._driver:
            self._driver.close()
            logger.info("Closed Neo4j driver")

    def verify_connectivity(self) -> None:
        """
        Verifies that the driver can connect to the database.

        Raises:
            ServiceUnavailable: If the database is not reachable.
        """
        try:
            self._driver.verify_connectivity()
            logger.info("Neo4j connectivity verified")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def execute_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Executes a Cypher query against the configured database.

        Args:
            query: The Cypher query string.
            parameters: A dictionary of parameters for the query.

        Returns:
            A list of dictionaries, where each dictionary represents a record.
        """
        if parameters is None:
            parameters = {}

        try:
            # We use execute_query which is the modern, recommended API in the 5.x+ driver
            # It handles retries and session management automatically.
            records, _, _ = self._driver.execute_query(
                query,
                parameters_=parameters,
                database_=self._database,
            )
            # Convert Neo4j Records to standard dicts
            # Record objects behave like dicts but we want pure python dicts for the return type
            return [r.data() for r in records]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
