# Product Requirements Document: coreason-graph-nexus

**Domain:** Knowledge Graph Construction, Ontology Alignment, & In-Memory Graph Reasoning
**Architectural Role:** The "Platinum Layer" Builder / The Graph Logic Engine. Hybrid Compute & ETL Edition
**Core Philosophy:** "Build robustly in Neo4j. Reason fast in NetworkX. Relations are First-Class Citizens."
**Dependencies:** coreason-mcp (Access), neo4j (Storage), networkx (Compute), scikit-learn (Clustering), redis (Cache)

## 1. Executive Summary

coreason-graph-nexus is the **Graph Orchestration & Reasoning Engine**.

It serves two distinct but connected functions:

1.  **The Builder (ETL):** It transforms static data (SQL, Parquet) into a connected semantic fabric stored in **Neo4j**. It uses **Universal Identity Resolution** to ensure "Tylenol" and "APAP" merge into a single node.
2.  **The Thinker (Compute):** It uses **networkx** for in-memory graph algorithms. Instead of relying on slow DB queries for complex logic, it projects subgraphs into memory to run algorithms like PageRank, Shortest Path, and Community Detection on the fly.

## 2. Functional Philosophy

The agent must implement the **Map-Resolve-Project-Compute Loop**:

1.  **Declarative Mapping:** We define "Projections" in YAML.
2.  **Universal Identity:** Before creation, nodes are resolved against the Master Ontology.
3.  **Performance Caching (Restored):** Ontology lookups are cached in **Redis** to ensure ingestion speed.
4.  **Hybrid Compute Architecture:**
    *   **Cold Storage (Neo4j):** Persistent storage for the global graph.
    *   **Hot Compute (NetworkX):** On-demand in-memory analysis of local neighborhoods.
5.  **Write-Back:** Algorithmic scores (e.g., Centrality) calculated in NetworkX are written back to Neo4j to enrich the permanent record.

## 3. Core Functional Requirements (Component Level)

### 3.1 The Projection Engine (The Builder)

**Concept:** A batch processor that reads from Silver/Gold layers and hydrates Neo4j.

*   **Mechanism:**
    *   **Initial Load:** Uses neo4j-admin import for massive CSV injection.
    *   **Delta Load:** Uses UNWIND Cypher batches (10k+ rows) for incremental updates.
*   **Validation:** Enforces schema constraints (e.g., "Drug must have an RxNorm ID").

### 3.2 The Ontology Resolver (The Librarian)

**Concept:** The authority on Naming Things.

*   **Integration:** connects to coreason-codex.
*   **Caching Strategy (Restored):**
    *   Check Redis for key "resolve:Tylenol".
    *   If Miss: Query codex, store result in Redis (TTL: 24h).
    *   *Value:* Reduces ETL latency from ms to µs.
*   **Action:** Maps source strings to Canonical Concept IDs.

### 3.3 The Graph Computer (The NetworkX Engine)

**Concept:** The in-memory logic core.

*   **Subgraph Projection:**
    *   Pulls a neighborhood (K-Hops) from Neo4j into a nx.DiGraph.
*   **Algorithmic Library:**
    *   **Pathfinding:** nx.all_simple_paths (MoA discovery).
    *   **Centrality:** nx.betweenness_centrality (KOL identification).
    *   **Clustering:** nx.community.louvain_communities (Topic detection).

### 3.4 The Link Predictor (The Analyst)

**Concept:** Infers implicit edges.

*   **Heuristic:** Rule-based (Author + Paper = WROTE).
*   **Semantic:** Uses Vector Embeddings. If Cosine(NodeA, NodeB) > Threshold, create SEMANTIC_LINK.

## 4. Integration Requirements

*   **Storage (neo4j):** Persistent Store.
*   **Cache (redis):** Ontology Lookup Cache.
*   **Compute (networkx):** Runtime Analysis.
*   **Access (coreason-mcp):** Exposes tools like graph_shortest_path to the LLM.

## 5. User Stories

### Story A: The "Mechanism of Action" (Pathfinding)

*   **Context:** "How does Drug X affect Protein Y?"
*   **Action:** Nexus loads the subgraph for X and Y into NetworkX and runs shortest_path.
*   **Result:** Returns the exact biological chain of interaction.

### Story B: The "Influence" Ranking (Centrality)

*   **Context:** "Who are the top experts?"
*   **Action:** Nexus runs PageRank on the Author-Paper graph in NetworkX.
*   **Write-Back:** Updates Neo4j Author nodes with pagerank_score.
*   **Result:** Future searches rank these authors higher.

### Story C: The "Synonym" Defense (Resolution)

*   **Context:** Source A says "Paracetamol", Source B says "Acetaminophen".
*   **Action:** Ontology Resolver maps both to RxNorm:161.
*   **Result:** Graph contains one node with two sources.

## 6. Data Schema

### ProjectionManifest (YAML)

```yaml
version: "2.1"
source_connection: "postgres://gold_db"
entities:
  - name: "Drug"
    source_table: "dim_products"
    ontology_strategy: "RxNorm"
relationships:
  - name: "TREATS"
    start_node: "Drug"
    end_node: "Disease"
```

### GraphJob (Operational - Restored)

```python
class GraphJob(BaseModel):
    id: UUID
    manifest_path: str
    status: Literal["RESOLVING", "PROJECTING", "COMPUTING", "COMPLETE"]
    metrics: dict = {
        "nodes_created": 0,
        "edges_created": 0,
        "ontology_cache_hits": 0
    }
```

### GraphAnalysisRequest (Runtime)

```python
class AnalysisAlgo(str, Enum):
    PAGERANK = "pagerank"
    SHORTEST_PATH = "shortest_path"

class GraphAnalysisRequest(BaseModel):
    center_node_id: str
    algorithm: AnalysisAlgo
    depth: int = 2
```

## 7. Implementation Directives for the Coding Agent

1.  **Batching is Mandatory:** Use UNWIND parameter maps in Cypher. Never execute single-row CREATE statements in a loop.
2.  **Redis Cache:** Implement a decorator @cached_resolver for the Ontology Resolver to hit Redis before Codex.
3.  **NetworkX Bridge:** Create a utility neo4j_to_networkx(driver, query) that efficiently transforms a Bolt Result stream into a NetworkX graph object.
4.  **Idempotency:** Use MERGE (Match or Create) for all node/edge ingestion to ensure jobs can be re-run safely.
