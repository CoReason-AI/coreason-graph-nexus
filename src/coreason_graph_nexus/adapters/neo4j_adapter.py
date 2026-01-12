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

import networkx as nx
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from neo4j.graph import Node, Path, Relationship

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

    def merge_nodes(
        self,
        label: str,
        data: list[dict[str, Any]],
        merge_keys: list[str],
        batch_size: int = 10000,
    ) -> None:
        """
        Ingests nodes using an idempotent MERGE operation in batches.

        Constructs a Cypher query that unwinds the data batch and MERGEs nodes
        based on the provided `merge_keys`. All other properties in the data
        dictionary are set on the node using `SET n += row`.

        Args:
            label: The primary label for the nodes (e.g., "Person").
            data: A list of dictionaries containing node properties.
            merge_keys: A list of property keys used to uniquely identify the node
                        (e.g., ["id"] or ["firstName", "lastName"]).
            batch_size: The number of records to process per transaction.
        """
        if not merge_keys:
            raise ValueError("merge_keys must not be empty.")

        # Construct the property map for the MERGE clause
        # e.g., { `id`: row.`id`, `name`: row.`name` }
        # We wrap keys in backticks to handle special characters.
        merge_props_str = ", ".join([f"`{key}`: row.`{key}`" for key in merge_keys])

        # Wrap label in backticks as well
        query = f"UNWIND $batch AS row MERGE (n:`{label}` {{ {merge_props_str} }}) SET n += row"

        logger.info(f"Merging nodes (Label: {label}) using keys: {merge_keys}")
        self.batch_write(query, data, batch_size=batch_size)

    def merge_relationships(
        self,
        start_label: str,
        start_data_key: str,
        end_label: str,
        end_data_key: str,
        rel_type: str,
        data: list[dict[str, Any]],
        start_node_prop: str = "id",
        end_node_prop: str = "id",
        batch_size: int = 10000,
    ) -> None:
        """
        Ingests relationships using an idempotent MERGE operation in batches.

        Constructs a Cypher query that unwinds the data batch, MATCHes the start
        and end nodes using their respective keys, and then MERGEs the relationship.
        Properties from the data dictionary are set on the relationship.

        Args:
            start_label: The label of the start node.
            start_data_key: The key in the input `data` dictionary that contains the start node identifier.
            end_label: The label of the end node.
            end_data_key: The key in the input `data` dictionary that contains the end node identifier.
            rel_type: The relationship type (e.g., "KNOWS").
            data: A list of dictionaries.
            start_node_prop: The property name on the start node to match against (default: "id").
            end_node_prop: The property name on the end node to match against (default: "id").
            batch_size: The number of records to process per transaction.
        """
        query = (
            f"UNWIND $batch AS row "
            f"MATCH (source:`{start_label}` {{ `{start_node_prop}`: row.`{start_data_key}` }}) "
            f"MATCH (target:`{end_label}` {{ `{end_node_prop}`: row.`{end_data_key}` }}) "
            f"MERGE (source)-[r:`{rel_type}`]->(target) "
            f"SET r += row"
        )

        logger.info(f"Merging relationships ({start_label})-[{rel_type}]->({end_label})")
        self.batch_write(query, data, batch_size=batch_size)

    def to_networkx(self, query: str, parameters: dict[str, Any] | None = None) -> nx.DiGraph:
        """
        Executes a Cypher query and converts the result into a NetworkX DiGraph.

        Parses Nodes, Relationships, and Paths from the raw Neo4j records.
        - Nodes are added with their element_id as the node ID, and properties + labels are stored.
        - Relationships are added as edges with type and properties.

        Args:
            query: The Cypher query string.
            parameters: A dictionary of parameters for the query.

        Returns:
            A networkx.DiGraph object representing the result subgraph.
        """
        if parameters is None:
            parameters = {}

        graph = nx.DiGraph()

        try:
            records, _, _ = self._driver.execute_query(
                query,
                parameters_=parameters,
                database_=self._database,
            )

            for record in records:
                for item in record.values():
                    if isinstance(item, Node):
                        # Use element_id (Neo4j 5.x) or id (Neo4j 4.x/compat)
                        # element_id is preferred for 5.x
                        node_id = item.element_id if hasattr(item, "element_id") else item.id
                        # Merge labels and properties
                        attrs = dict(item.items())
                        attrs["labels"] = list(item.labels)
                        graph.add_node(node_id, **attrs)

                    elif isinstance(item, Relationship):
                        start_id = (
                            item.start_node.element_id if hasattr(item.start_node, "element_id") else item.start_node.id
                        )
                        end_id = item.end_node.element_id if hasattr(item.end_node, "element_id") else item.end_node.id
                        attrs = dict(item.items())
                        attrs["type"] = item.type
                        graph.add_edge(start_id, end_id, **attrs)

                    elif isinstance(item, Path):
                        # Iterate over nodes and relationships in the path
                        for node in item.nodes:
                            node_id = node.element_id if hasattr(node, "element_id") else node.id
                            attrs = dict(node.items())
                            attrs["labels"] = list(node.labels)
                            graph.add_node(node_id, **attrs)

                        for rel in item.relationships:
                            start_id = (
                                rel.start_node.element_id
                                if hasattr(rel.start_node, "element_id")
                                else rel.start_node.id
                            )
                            end_id = rel.end_node.element_id if hasattr(rel.end_node, "element_id") else rel.end_node.id
                            attrs = dict(rel.items())
                            attrs["type"] = rel.type
                            graph.add_edge(start_id, end_id, **attrs)
                    elif isinstance(item, list):
                        # Handle list of things (e.g. collected nodes)
                        for subitem in item:
                            if isinstance(subitem, Node):
                                node_id = subitem.element_id if hasattr(subitem, "element_id") else subitem.id
                                attrs = dict(subitem.items())
                                attrs["labels"] = list(subitem.labels)
                                graph.add_node(node_id, **attrs)
                            elif isinstance(subitem, Relationship):
                                start_id = (
                                    subitem.start_node.element_id
                                    if hasattr(subitem.start_node, "element_id")
                                    else subitem.start_node.id
                                )
                                end_id = (
                                    subitem.end_node.element_id
                                    if hasattr(subitem.end_node, "element_id")
                                    else subitem.end_node.id
                                )
                                attrs = dict(subitem.items())
                                attrs["type"] = subitem.type
                                graph.add_edge(start_id, end_id, **attrs)

            logger.info(
                f"Converted Cypher result to NetworkX graph: "
                f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            return graph

        except Exception as e:
            logger.error(f"Failed to convert Cypher to NetworkX: {e}")
            raise
