# coreason-graph-nexus

**Graph Orchestration & Reasoning Engine**

The "Platinum Layer" Builder and Graph Logic Engine. A hybrid system combining Neo4j for robust storage and NetworkX for fast, in-memory reasoning.

[![License](https://img.shields.io/badge/License-Prosperity%203.0-blue.svg)](https://github.com/CoReason-AI/coreason_graph_nexus/blob/main/LICENSE)
[![CI](https://github.com/CoReason-AI/coreason_graph_nexus/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_graph_nexus/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Installation

```bash
pip install coreason-graph-nexus
```

## Features

**1. The Builder (ETL)**
   - Transforms static data (SQL, Parquet) into a connected semantic fabric in Neo4j.
   - **Universal Identity Resolution:** Maps source terms (e.g., "Tylenol") to canonical concepts (e.g., "RxNorm:161") using a cached ontology resolver.
   - **Performance:** Utilizes `UNWIND` Cypher batches and Redis caching for high-speed ingestion.

**2. The Thinker (Compute)**
   - **Hybrid Architecture:** Uses Neo4j for cold storage and NetworkX for hot, in-memory compute.
   - **Graph Algorithms:** Projects subgraphs to run PageRank, Shortest Path, and Community Detection (Louvain) on the fly.
   - **Write-Back:** Persists algorithmic scores back to Neo4j.

**3. The Analyst (Link Prediction)**
   - **Heuristic:** Executes rule-based predictions via Cypher queries.
   - **Semantic:** Infers implicit links using vector embeddings and cosine similarity.

## Usage

```python
from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient
from coreason_graph_nexus.compute import GraphComputer
from coreason_graph_nexus.models import GraphAnalysisRequest, AnalysisAlgo

# 1. Initialize Connection
client = Neo4jClient(uri="bolt://localhost:7687", auth=("neo4j", "password"))

# 2. Run Graph Analysis (The Thinker)
computer = GraphComputer(client)

# Request PageRank on a subgraph centered at a node with depth 2
request = GraphAnalysisRequest(
    center_node_id="RxNorm:161",
    algorithm=AnalysisAlgo.PAGERANK,
    depth=2
)

with client:
    # Projects subgraph to NetworkX -> Computes PageRank -> Writes back to Neo4j
    results = computer.run_analysis(request)
    print(f"PageRank Scores: {results}")
```
