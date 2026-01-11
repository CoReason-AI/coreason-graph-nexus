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

    def batch_write(
        self,
        query: str,
        data: list[dict[str, Any]],
        batch_size: int = 10000,
        batch_param_name: str = "batch",
    ) -> None:
        """
        Executes a Cypher query in batches.

        This method is designed for high-throughput ingestion using the UNWIND pattern.
        The data list is sliced into chunks, and each chunk is passed as a parameter
        to the Cypher query.

        Args:
            query: The Cypher query string. It should typically start with
                   'UNWIND $batch AS row ...' (where 'batch' matches batch_param_name).
            data: A list of dictionaries to be ingested.
            batch_size: The number of records to process in a single transaction.
            batch_param_name: The key used in the parameters dictionary for the list.
        """
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        if not data:
            logger.info("No data provided for batch write.")
            return

        total = len(data)
        logger.info(f"Starting batch write: {total} records, batch_size={batch_size}")

        for i in range(0, total, batch_size):
            chunk = data[i : i + batch_size]
            try:
                self.execute_query(query, parameters={batch_param_name: chunk})
            except Exception as e:
                logger.error(f"Batch write failed at index {i} (processing {len(chunk)} records): {e}")
                raise

        logger.info(f"Batch write completed: {total} records processed.")
