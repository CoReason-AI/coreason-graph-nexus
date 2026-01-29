# The Architecture and Utility of coreason-graph-nexus

### 1. The Philosophy (The Why)

Standard ETL pipelines treat data as inert cargo—moving tables from Point A to Point B without understanding them. **coreason-graph-nexus** rejects this passivity. It is an **Orchestration & Reasoning Engine** designed to transform disconnected static records into a living, semantic fabric.

The architecture solves a critical "split-brain" problem in modern knowledge graphs: storage databases (like Neo4j) are robust but slow at complex iterative math, while compute libraries (like NetworkX) are brilliant but ephemeral. `coreason-graph-nexus` bridges this gap. It acts as the "Platinum Layer" builder, employing a **Map-Resolve-Project-Compute** loop. It doesn't just load data; it resolves identity (knowing that "Tylenol" and "Acetaminophen" are the same node) and calculates implicit connections on the fly. By treating relationships as first-class citizens and leveraging a hybrid compute model—cold storage for persistence, hot memory for logic—it turns a graph database into a reasoning engine.

### 2. Under the Hood (The Dependencies & logic)

The package constructs a best-in-class stack to handle the specific lifecycle of knowledge graph data:

*   **`neo4j` (The Storage Cortex):** Serves as the persistent source of truth. The engine uses it for what databases do best: storage, retrieval, and ACID-compliant transactional writes.
*   **`networkx` (The Thinker):** When deep reasoning is required—like calculating PageRank, Betweenness Centrality, or finding complex paths—the engine projects subgraphs into memory here. This avoids punishing the database with heavy iterative queries.
*   **`redis` (The Librarian):** Identity resolution is expensive. To keep ingestion blazing fast, the `OntologyResolver` uses Redis to cache lookups, ensuring that repeated terms don't trigger redundant external API calls.
*   **`scikit-learn` & `numpy` (The Analyst):** Powers the **Semantic Link Predictor**. By leveraging vector embeddings and cosine similarity, it allows the graph to "dream" new edges between nodes that are conceptually similar but explicitly unconnected.
*   **`fastapi` & `uvicorn` (The Server):** Exposes the engine as a high-performance **Microservice (Service G)**, allowing external systems to trigger ingestion, analysis, and prediction via REST.
*   **`pydantic`:** Enforces strict schema validation on the "Manifests" (declarative ETL configs), preventing bad data from ever entering the graph.

The internal logic is driven by a `ProjectionEngine` that uses sophisticated closure-based batching (via `UNWIND` Cypher clauses) to handle massive datasets efficiently, and a `GraphComputer` that orchestrates the projection-computation-writeback cycle seamlessly.

### 3. In Practice (The How)

The following examples demonstrate the "Happy Path"—ingesting data, reasoning about it, and discovering new links.

**The Builder: Ingesting Data with Identity Resolution**
The `ProjectionEngine` reads from a source, resolves identities against an ontology, and projects nodes into Neo4j using efficient batching.

```python
from coreason_graph_nexus.projector import ProjectionEngine
from coreason_graph_nexus.models import GraphJob, ProjectionManifest

# Initialize the engine
engine = ProjectionEngine(client=neo4j_client, resolver=ontology_resolver)

# Define the job (The "What")
manifest = ProjectionManifest.from_yaml("manifests/drug_safety.yaml")
job = GraphJob(id=uuid4(), manifest_path="drug_safety.yaml", status="RESOLVING")

# Execute ingestion (The "How")
# This automatically handles batching, caching, and ontology resolution
engine.ingest_entities(manifest, source_adapter, job)
engine.ingest_relationships(manifest, source_adapter, job)
```

**The Thinker: In-Memory Reasoning**
Instead of writing complex Cypher queries, the `GraphComputer` pulls a subgraph into memory to run standard NetworkX algorithms, then writes the scores back to the database.

```python
from coreason_graph_nexus.compute import GraphComputer
from coreason_graph_nexus.models import GraphAnalysisRequest, AnalysisAlgo

computer = GraphComputer(client=neo4j_client)

# Request PageRank to find Key Opinion Leaders (KOLs)
request = GraphAnalysisRequest(
    center_node_id="Author:12345",
    algorithm=AnalysisAlgo.PAGERANK,
    depth=2,
    write_property="pagerank_score"
)

# Run analysis: Fetch -> Compute (NetworkX) -> Write Back (Neo4j)
scores = computer.run_analysis(request)
```

**The Analyst: Semantic Link Prediction**
The `LinkPredictor` finds hidden connections using vector embeddings, creating "SEMANTIC_LINK" edges between nodes that are mathematically similar.

```python
from coreason_graph_nexus.link_prediction import LinkPredictor
from coreason_graph_nexus.models import LinkPredictionRequest, LinkPredictionMethod

predictor = LinkPredictor(client=neo4j_client)

# Find drugs that are semantically similar to a specific phenotype
request = LinkPredictionRequest(
    method=LinkPredictionMethod.SEMANTIC,
    source_label="Drug",
    target_label="Phenotype",
    threshold=0.85,  # High similarity only
    relationship_type="POTENTIAL_TREATMENT"
)

# Compute similarity matrix and materialize new edges
predictor.predict_links(request)
```

### 4. Service G: The Graph Logic Microservice

You can run `coreason-graph-nexus` as a standalone microservice using Docker or Uvicorn.

**Start the Server:**
```bash
uvicorn coreason_graph_nexus.server:app --host 0.0.0.0 --port 8000
```

**Trigger Ingestion (Async):**
```bash
curl -X POST "http://localhost:8000/project/ingest" \
     -H "Content-Type: application/json" \
     -d '{
           "manifest": {
             "version": "1.0",
             "source_connection": "...",
             "entities": [...],
             "relationships": [...]
           },
           "source_base_path": "/data"
         }'
```

**Request Analysis (PageRank):**
```bash
curl -X POST "http://localhost:8000/compute/analysis" \
     -H "Content-Type: application/json" \
     -d '{
           "center_node_id": "Node:1",
           "algorithm": "pagerank",
           "depth": 2
         }'
```
