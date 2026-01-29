# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from typing import Any, cast

import anyio
import networkx as nx

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient, Neo4jClientAsync
from coreason_graph_nexus.models import AnalysisAlgo, GraphAnalysisRequest
from coreason_graph_nexus.utils.logger import logger


class GraphComputer:
    """
    The Graph Logic Engine ("The Thinker").

    Responsible for:
    1. Projecting subgraphs from Neo4j into memory (NetworkX).
    2. Running graph algorithms (PageRank, Shortest Path, Louvain).
    3. Writing results back to Neo4j.
    """

    def __init__(self, client: Neo4jClient) -> None:
        """
        Initialize the GraphComputer.

        Args:
            client: An initialized Neo4jClient.
        """
        self.client = client

    def run_analysis(self, request: GraphAnalysisRequest) -> Any:
        """
        Executes the requested graph analysis algorithm.

        Args:
            request: The analysis request configuration.

        Returns:
            The result of the analysis (structure depends on the algorithm).

        Raises:
            ValueError: If required parameters for specific algorithms are missing.
            NotImplementedError: If the requested algorithm is not implemented.
        """
        logger.info(
            f"Starting analysis: {request.algorithm.value} (center={request.center_node_id}, depth={request.depth})"
        )

        # 1. Fetch Subgraph
        graph = self._fetch_subgraph(request.center_node_id, request.depth)
        if graph.number_of_nodes() == 0:
            logger.warning(f"Subgraph is empty for center_node_id={request.center_node_id}")

        # 2. Run Algorithm
        if request.algorithm == AnalysisAlgo.PAGERANK:
            return self._compute_pagerank(graph, request.write_property)
        elif request.algorithm == AnalysisAlgo.SHORTEST_PATH:
            if not request.target_node_id:
                raise ValueError("target_node_id is required for SHORTEST_PATH algorithm")
            return self._compute_shortest_path(graph, request.center_node_id, request.target_node_id)
        elif request.algorithm == AnalysisAlgo.LOUVAIN:
            return self._compute_louvain(graph, request.write_property)
        else:
            raise NotImplementedError(f"Algorithm {request.algorithm} is not implemented")

    def _fetch_subgraph(self, center_id: str, depth: int) -> nx.DiGraph:
        """
        Fetches a K-Hop subgraph centered around a specific node.

        Uses standard Cypher variable-length path matching.

        Args:
            center_id: The ID (property `id` or `elementId`) of the center node.
            depth: The number of hops.

        Returns:
            A NetworkX DiGraph representing the subgraph.
        """
        # We try to match by property 'id' first (business key), then fallback/OR to elementId checks if needed?
        # Ideally, center_id corresponds to the 'id' property stored on nodes.
        # Constructing a path query up to 'depth' hops.
        # Note: Cypher doesn't allow parameters for variable length depth (e.g. *..$depth).
        # We must inject it safely. depth is validated as int in model.

        query = f"MATCH path = (n)-[*..{depth}]-(m) WHERE n.id = $center_id OR elementId(n) = $center_id RETURN path"
        logger.debug(f"Fetching subgraph with query: {query}")
        return self.client.to_networkx(query, parameters={"center_id": center_id})

    def _compute_pagerank(self, graph: nx.DiGraph, write_property: str) -> dict[str, float]:
        """
        Computes PageRank and writes scores back to Neo4j.

        Args:
            graph: The in-memory graph.
            write_property: The property key to update in Neo4j.

        Returns:
            A dictionary mapping node IDs to PageRank scores.
        """
        if graph.number_of_nodes() == 0:
            return {}

        scores = nx.pagerank(graph)
        logger.info(f"Computed PageRank for {len(scores)} nodes")

        # Prepare write-back data
        # keys in scores are node IDs (element_id or id from to_networkx)
        # We write back matching on elementId (assuming to_networkx returned element_ids as keys)
        # If keys are business IDs, the query below needs to be adjusted.
        # Neo4jClient.to_networkx uses element_id if available.
        # So we match on elementId(n).

        data = [{"id": k, "value": v} for k, v in scores.items()]

        # Write back
        # We use a custom batch write query
        # We match using elementId first, if not found, maybe match on id?
        # Actually, if we got the ID from Neo4j, we should use the same mechanism to match it back.
        # The to_networkx uses element_id. So we use elementId lookup.

        query = f"UNWIND $batch AS row MATCH (n) WHERE elementId(n) = row.id SET n.`{write_property}` = row.value"

        self.client.batch_write(query, data)
        return cast(dict[str, float], scores)

    def _compute_shortest_path(self, graph: nx.DiGraph, source: str, target: str) -> list[str]:
        """
        Computes shortest path between source and target in the subgraph.

        Args:
            graph: The subgraph.
            source: Source node ID (as passed in request - business ID).
            target: Target node ID (as passed in request - business ID).

        Returns:
            List of node IDs in the path.

        Raises:
            ValueError: If source or target node is not found in the subgraph.
        """
        # Problem: 'source' and 'target' strings passed here are likely business IDs (e.g., 'RxNorm:123').
        # The 'graph' nodes are keyed by Neo4j element_id (internal ID) because of to_networkx.
        # We need to map business IDs to internal graph node IDs to run nx.shortest_path.

        # 1. Build a map of property_id -> graph_node_id
        # We look at node attributes in the graph.
        id_map = {}
        for n, attrs in graph.nodes(data=True):
            if "id" in attrs:
                id_map[attrs["id"]] = n
            # Also map element_id to itself just in case
            id_map[str(n)] = n

        # 2. Resolve source/target
        source_node = id_map.get(source)
        target_node = id_map.get(target)

        if not source_node:
            raise ValueError(f"Source node '{source}' not found in the subgraph.")
        if not target_node:
            raise ValueError(f"Target node '{target}' not found in the subgraph.")

        try:
            path = nx.shortest_path(graph, source=source_node, target=target_node)
            logger.info(f"Found shortest path length: {len(path)}")
            return path  # type: ignore # nx returns list
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {source} and {target}")
            return []

    def _compute_louvain(self, graph: nx.DiGraph, write_property: str) -> dict[str, int]:
        """
        Computes Louvain communities and writes ID back to Neo4j.

        Args:
            graph: The in-memory graph.
            write_property: The property key to update in Neo4j.

        Returns:
            A dictionary mapping node IDs to community IDs.
        """
        if graph.number_of_nodes() == 0:
            return {}

        # Louvain works on undirected graphs usually, but networkx implementation handles DiGraph
        # by converting to undirected or using specific directed algo?
        # nx.community.louvain_communities works on undirected graphs.
        # We convert to undirected view for community detection.
        undirected_graph = graph.to_undirected()

        communities = nx.community.louvain_communities(undirected_graph)
        # communities is list[set[node_id]]

        result = {}
        data = []
        for idx, community in enumerate(communities):
            for node_id in community:
                result[node_id] = idx
                data.append({"id": node_id, "value": idx})

        logger.info(f"Detected {len(communities)} communities")

        query = f"UNWIND $batch AS row MATCH (n) WHERE elementId(n) = row.id SET n.`{write_property}` = row.value"
        self.client.batch_write(query, data)
        return result


class GraphComputerAsync:
    """
    The Graph Logic Engine ("The Thinker") - Async Version.

    Responsible for:
    1. Projecting subgraphs from Neo4j into memory (NetworkX).
    2. Running graph algorithms (PageRank, Shortest Path, Louvain) in threads.
    3. Writing results back to Neo4j.
    """

    def __init__(self, client: Neo4jClientAsync) -> None:
        """
        Initialize the GraphComputerAsync.

        Args:
            client: An initialized Neo4jClientAsync.
        """
        self.client = client

    async def run_analysis(self, request: GraphAnalysisRequest) -> Any:
        """
        Executes the requested graph analysis algorithm.

        Args:
            request: The analysis request configuration.

        Returns:
            The result of the analysis (structure depends on the algorithm).
        """
        logger.info(
            f"Starting analysis: {request.algorithm.value} (center={request.center_node_id}, depth={request.depth})"
        )

        # 1. Fetch Subgraph (IO bound)
        graph = await self._fetch_subgraph(request.center_node_id, request.depth)
        if graph.number_of_nodes() == 0:
            logger.warning(f"Subgraph is empty for center_node_id={request.center_node_id}")

        # 2. Run Algorithm (CPU bound -> Thread) & Write back (IO bound)
        if request.algorithm == AnalysisAlgo.PAGERANK:
            return await self._compute_pagerank(graph, request.write_property)
        elif request.algorithm == AnalysisAlgo.SHORTEST_PATH:
            if not request.target_node_id:
                raise ValueError("target_node_id is required for SHORTEST_PATH algorithm")
            return await self._compute_shortest_path(graph, request.center_node_id, request.target_node_id)
        elif request.algorithm == AnalysisAlgo.LOUVAIN:
            return await self._compute_louvain(graph, request.write_property)
        else:
            raise NotImplementedError(f"Algorithm {request.algorithm} is not implemented")

    async def _fetch_subgraph(self, center_id: str, depth: int) -> nx.DiGraph:
        query = f"MATCH path = (n)-[*..{depth}]-(m) WHERE n.id = $center_id OR elementId(n) = $center_id RETURN path"
        logger.debug(f"Fetching subgraph with query: {query}")
        return await self.client.to_networkx(query, parameters={"center_id": center_id})

    async def _compute_pagerank(self, graph: nx.DiGraph, write_property: str) -> dict[str, float]:
        if graph.number_of_nodes() == 0:
            return {}

        # Offload CPU compute
        scores = await anyio.to_thread.run_sync(nx.pagerank, graph)
        logger.info(f"Computed PageRank for {len(scores)} nodes")

        data = [{"id": k, "value": v} for k, v in scores.items()]

        query = f"UNWIND $batch AS row MATCH (n) WHERE elementId(n) = row.id SET n.`{write_property}` = row.value"

        await self.client.batch_write(query, data)
        return cast(dict[str, float], scores)

    async def _compute_shortest_path(self, graph: nx.DiGraph, source: str, target: str) -> list[str]:
        # Helper to run in thread
        def find_path() -> list[str]:
            id_map = {}
            for n, attrs in graph.nodes(data=True):
                if "id" in attrs:
                    id_map[attrs["id"]] = n
                id_map[str(n)] = n

            source_node = id_map.get(source)
            target_node = id_map.get(target)

            if not source_node:
                raise ValueError(f"Source node '{source}' not found in the subgraph.")
            if not target_node:
                raise ValueError(f"Target node '{target}' not found in the subgraph.")

            try:
                path = nx.shortest_path(graph, source=source_node, target=target_node)
                logger.info(f"Found shortest path length: {len(path)}")
                return path  # type: ignore
            except nx.NetworkXNoPath:
                logger.warning(f"No path found between {source} and {target}")
                return []

        return cast(list[str], await anyio.to_thread.run_sync(find_path))

    async def _compute_louvain(self, graph: nx.DiGraph, write_property: str) -> dict[str, int]:
        if graph.number_of_nodes() == 0:
            return {}

        # Helper for CPU compute
        def run_louvain() -> tuple[dict[str, int], list[dict[str, Any]]]:
            undirected_graph = graph.to_undirected()
            communities = nx.community.louvain_communities(undirected_graph)

            result = {}
            data = []
            for idx, community in enumerate(communities):
                for node_id in community:
                    result[node_id] = idx
                    data.append({"id": node_id, "value": idx})
            return result, data

        result, data = await anyio.to_thread.run_sync(run_louvain)
        logger.info(f"Detected {len(set(result.values()))} communities")

        query = f"UNWIND $batch AS row MATCH (n) WHERE elementId(n) = row.id SET n.`{write_property}` = row.value"
        await self.client.batch_write(query, data)
        return cast(dict[str, int], result)
