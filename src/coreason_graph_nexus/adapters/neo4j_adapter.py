# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from collections.abc import Iterable
from itertools import batched
from types import TracebackType
from typing import Any, Self, cast

import networkx as nx
from neo4j import AsyncGraphDatabase, GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from neo4j.graph import Node, Path, Relationship

from coreason_graph_nexus.config import settings
from coreason_graph_nexus.utils.logger import logger


class Neo4jClient:
    """
    A client wrapper for the Neo4j Graph Database (Sync).
    """

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str],
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._auth = auth
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=auth)
        logger.info(f"Initialized Neo4j driver (Sync) for {uri} (db: {database})")

    def __enter__(self) -> Self:
        self.verify_connectivity()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            logger.info("Closed Neo4j driver (Sync)")

    def verify_connectivity(self) -> None:
        try:
            self._driver.verify_connectivity()
            logger.info("Neo4j connectivity verified (Sync)")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def execute_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if parameters is None:
            parameters = {}
        try:
            records, _, _ = self._driver.execute_query(
                query,
                parameters_=parameters,
                database_=self._database,
            )
            return [r.data() for r in records]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def batch_write(
        self,
        query: str,
        data: Iterable[dict[str, Any]],
        batch_size: int = settings.default_batch_size,
        batch_param_name: str = "batch",
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        count = 0
        for chunk in batched(data, batch_size):
            chunk_list = list(chunk)
            try:
                self.execute_query(query, parameters={batch_param_name: chunk_list})
                count += len(chunk_list)
            except Exception as e:
                logger.error(f"Batch write failed after processing {count} records: {e}")
                raise

    def merge_nodes(
        self,
        label: str,
        data: Iterable[dict[str, Any]],
        merge_keys: list[str],
        batch_size: int = settings.default_batch_size,
    ) -> None:
        if not merge_keys:
            raise ValueError("merge_keys must not be empty.")
        merge_props_str = ", ".join([f"`{key}`: row.`{key}`" for key in merge_keys])
        query = f"UNWIND $batch AS row MERGE (n:`{label}` {{ {merge_props_str} }}) SET n += row"
        self.batch_write(query, data, batch_size=batch_size)

    def merge_relationships(
        self,
        start_label: str,
        start_data_key: str,
        end_label: str,
        end_data_key: str,
        rel_type: str,
        data: Iterable[dict[str, Any]],
        start_node_prop: str = "id",
        end_node_prop: str = "id",
        batch_size: int = settings.default_batch_size,
    ) -> None:
        query = (
            f"UNWIND $batch AS row "
            f"MATCH (source:`{start_label}` {{ `{start_node_prop}`: row.`{start_data_key}` }}) "
            f"MATCH (target:`{end_label}` {{ `{end_node_prop}`: row.`{end_data_key}` }}) "
            f"MERGE (source)-[r:`{rel_type}`]->(target) "
            f"SET r += row"
        )
        self.batch_write(query, data, batch_size=batch_size)

    def to_networkx(self, query: str, parameters: dict[str, Any] | None = None) -> nx.DiGraph:
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
                    _process_graph_item(graph, item)
            return graph
        except Exception as e:
            logger.error(f"Failed to convert Cypher to NetworkX: {e}")
            raise


class Neo4jClientAsync:
    """
    A client wrapper for the Neo4j Graph Database (Async).

    This client manages the driver lifecycle and provides a simplified interface
    for executing Cypher queries using best practices (e.g., connection pooling).
    It implements the Async Context Manager protocol.
    """

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str],
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._auth = auth
        self._database = database
        self._driver = AsyncGraphDatabase.driver(uri, auth=auth)
        logger.info(f"Initialized Neo4j driver (Async) for {uri} (db: {database})")

    async def __aenter__(self) -> Self:
        await self.verify_connectivity()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            logger.info("Closed Neo4j driver (Async)")

    async def verify_connectivity(self) -> None:
        try:
            await self._driver.verify_connectivity()
            logger.info("Neo4j connectivity verified (Async)")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def execute_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if parameters is None:
            parameters = {}
        try:
            records, _, _ = await self._driver.execute_query(
                query,
                parameters_=parameters,
                database_=self._database,
            )
            return [r.data() for r in records]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    async def batch_write(
        self,
        query: str,
        data: Iterable[dict[str, Any]],
        batch_size: int = settings.default_batch_size,
        batch_param_name: str = "batch",
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

        count = 0
        for chunk in batched(data, batch_size):
            chunk_list = list(chunk)
            try:
                await self.execute_query(query, parameters={batch_param_name: chunk_list})
                count += len(chunk_list)
            except Exception as e:
                logger.error(f"Batch write failed after processing {count} records: {e}")
                raise

    async def merge_nodes(
        self,
        label: str,
        data: Iterable[dict[str, Any]],
        merge_keys: list[str],
        batch_size: int = settings.default_batch_size,
    ) -> None:
        if not merge_keys:
            raise ValueError("merge_keys must not be empty.")
        merge_props_str = ", ".join([f"`{key}`: row.`{key}`" for key in merge_keys])
        query = f"UNWIND $batch AS row MERGE (n:`{label}` {{ {merge_props_str} }}) SET n += row"
        await self.batch_write(query, data, batch_size=batch_size)

    async def merge_relationships(
        self,
        start_label: str,
        start_data_key: str,
        end_label: str,
        end_data_key: str,
        rel_type: str,
        data: Iterable[dict[str, Any]],
        start_node_prop: str = "id",
        end_node_prop: str = "id",
        batch_size: int = settings.default_batch_size,
    ) -> None:
        query = (
            f"UNWIND $batch AS row "
            f"MATCH (source:`{start_label}` {{ `{start_node_prop}`: row.`{start_data_key}` }}) "
            f"MATCH (target:`{end_label}` {{ `{end_node_prop}`: row.`{end_data_key}` }}) "
            f"MERGE (source)-[r:`{rel_type}`]->(target) "
            f"SET r += row"
        )
        await self.batch_write(query, data, batch_size=batch_size)

    async def to_networkx(self, query: str, parameters: dict[str, Any] | None = None) -> nx.DiGraph:
        if parameters is None:
            parameters = {}
        graph = nx.DiGraph()
        try:
            records, _, _ = await self._driver.execute_query(
                query,
                parameters_=parameters,
                database_=self._database,
            )
            for record in records:
                for item in record.values():
                    _process_graph_item(graph, item)
            return graph
        except Exception as e:
            logger.error(f"Failed to convert Cypher to NetworkX: {e}")
            raise


# Shared Helper Functions for Graph Conversion


def _process_graph_item(graph: nx.DiGraph, item: Any) -> None:
    if isinstance(item, Node):
        _add_node_to_graph(graph, item)
    elif isinstance(item, Relationship):
        _add_relationship_to_graph(graph, item)
    elif isinstance(item, Path):
        for node in item.nodes:
            _add_node_to_graph(graph, node)
        for rel in item.relationships:
            _add_relationship_to_graph(graph, rel)
    elif isinstance(item, list):
        for subitem in item:
            _process_graph_item(graph, subitem)


def _add_node_to_graph(graph: nx.DiGraph, node: Node) -> None:
    node_id = _get_node_id(node)
    attrs = dict(node.items())
    attrs["labels"] = list(node.labels)
    graph.add_node(node_id, **attrs)


def _add_relationship_to_graph(graph: nx.DiGraph, rel: Relationship) -> None:
    start_id = _get_node_id(rel.start_node)
    end_id = _get_node_id(rel.end_node)
    attrs = dict(rel.items())
    attrs["type"] = rel.type
    graph.add_edge(start_id, end_id, **attrs)


def _get_node_id(node: Node) -> str | int:
    if hasattr(node, "element_id"):
        return str(node.element_id)
    return cast(str | int, node.id)
