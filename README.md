# coreason-graph-nexus

> **The Graph Orchestration & Reasoning Engine**

[![Organization](https://img.shields.io/badge/org-CoReason--AI-blue)](https://github.com/CoReason-AI)
[![License: Prosperity 3.0](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![CI](https://github.com/CoReason-AI/coreason-graph-nexus/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason-graph-nexus/actions)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-Product%20Requirements-green)](docs/product_requirements.md)

**coreason-graph-nexus** acts as the "Platinum Layer" Builder and Graph Logic Engine. It bridges the gap between persistent graph storage (Neo4j) and high-speed in-memory reasoning (NetworkX), ensuring robust data ingestion, ontology alignment, and advanced algorithmic analysis.

---

## 🚀 Features

*   **Hybrid Compute Architecture:** Seamlessly moves data between Cold Storage (Neo4j) and Hot Compute (NetworkX) for on-demand analysis.
*   **Projection Engine (The Builder):** Declarative ETL pipeline that transforms raw data into a semantic graph, enforcing schema constraints.
*   **Ontology Resolver (The Librarian):**  Standardizes entity identities (e.g., merging "Tylenol" and "APAP") with high-performance Redis caching.
*   **Graph Computer (The Thinker):** Executes complex algorithms like PageRank, Betweenness Centrality, and Community Detection in memory.
*   **Link Predictor (The Analyst):** Infers implicit relationships using both heuristic rules and semantic vector embeddings.
*   **Graph Logic Microservice (Service G):** Exposes all engine capabilities via a high-performance REST API (FastAPI) for seamless integration into microservice architectures.

## 📦 Installation

```bash
pip install coreason-graph-nexus
```

## 🛠️ Usage

Here is a quick example of how to initialize the client and run a PageRank analysis:

```python
from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient
from coreason_graph_nexus.compute import GraphComputer
from coreason_graph_nexus.models import GraphAnalysisRequest, AnalysisAlgo

# 1. Initialize Connection
neo4j_auth = ("neo4j", "password")
with Neo4jClient(uri="bolt://localhost:7687", auth=neo4j_auth) as client:

    # 2. Initialize the Graph Computer
    computer = GraphComputer(client)

    # 3. Define Analysis Request
    request = GraphAnalysisRequest(
        center_node_id="RxNorm:123",
        algorithm=AnalysisAlgo.PAGERANK,
        depth=2,
        write_property="pagerank_score"
    )

    # 4. Run Analysis
    results = computer.run_analysis(request)
    print(f"Computed PageRank for {len(results)} nodes.")
```

### Running as a Microservice

You can also deploy the engine as a standalone service:

```bash
# Start the API server
uvicorn coreason_graph_nexus.server:app --host 0.0.0.0 --port 8000
```

Refer to [`vignette.md`](vignette.md) for detailed API usage examples.

## 📄 License

This software is dual-licensed. It is available under the **Prosperity Public License 3.0** for open-source and non-commercial use. Commercial use beyond a 30-day trial requires a separate license.

Copyright (c) 2025 CoReason, Inc.
