# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import UUID

import yaml
from pydantic import BaseModel, Field, model_validator


class PropertyMapping(BaseModel):
    """
    Maps a column in the source table to a property on the graph node.

    Attributes:
        source: The column name in the source table.
        target: The property name in the graph node.
    """

    source: str = Field(min_length=1, description="The column name in the source table.")
    target: str = Field(min_length=1, description="The property name in the graph node.")


class Entity(BaseModel):
    """
    Defines how to map a source table to a Graph Node (Entity).

    Attributes:
        name: The label of the node (e.g., 'Drug').
        source_table: The database table or view to read from.
        id_column: The column that uniquely identifies the entity.
        ontology_mapping: The ontology strategy to use for resolution (e.g., 'RxNorm').
        properties: A list of property mappings.
    """

    name: str = Field(min_length=1, description="The label of the node (e.g., 'Drug').")
    source_table: str = Field(min_length=1, description="The database table or view to read from.")
    id_column: str = Field(min_length=1, description="The column that uniquely identifies the entity.")
    ontology_mapping: str = Field(min_length=1, description="The ontology strategy to use for resolution.")
    properties: list[PropertyMapping] = Field(description="A list of property mappings.")

    @model_validator(mode="after")
    def validate_unique_property_targets(self) -> "Entity":
        """
        Validates that no two properties map to the same target field.
        """
        targets = [p.target for p in self.properties]
        if len(targets) != len(set(targets)):
            seen = set()
            duplicates = set()
            for x in targets:
                if x in seen:
                    duplicates.add(x)
                seen.add(x)
            raise ValueError(f"Duplicate property targets found in Entity '{self.name}': {', '.join(duplicates)}")
        return self


class Relationship(BaseModel):
    """
    Defines how to map a source table to a Graph Relationship (Edge).

    Attributes:
        name: The type of the relationship (e.g., 'REPORTED_EVENT').
        source_table: The database table or view to read from.
        start_node: The label of the starting node (must match an Entity name).
        start_key: The foreign key column in the source table pointing to the start node.
        end_node: The label of the ending node (must match an Entity name).
        end_key: The foreign key column in the source table pointing to the end node.
    """

    name: str = Field(min_length=1, description="The type of the relationship (e.g., 'REPORTED_EVENT').")
    source_table: str = Field(min_length=1, description="The database table or view to read from.")
    start_node: str = Field(min_length=1, description="The label of the starting node (must match an Entity name).")
    start_key: str = Field(
        min_length=1, description="The foreign key column in the source table pointing to the start node."
    )
    end_node: str = Field(min_length=1, description="The label of the ending node (must match an Entity name).")
    end_key: str = Field(
        min_length=1, description="The foreign key column in the source table pointing to the end node."
    )


class ProjectionManifest(BaseModel):
    """
    The master configuration for a Graph Projection job.

    Attributes:
        version: The version of the manifest schema.
        source_connection: The connection string for the source database.
        entities: A list of Entity definitions.
        relationships: A list of Relationship definitions.
    """

    version: str = Field(min_length=1, description="The version of the manifest schema.")
    source_connection: str = Field(min_length=1, description="The connection string for the source database.")
    entities: list[Entity] = Field(description="A list of Entity definitions.")
    relationships: list[Relationship] = Field(description="A list of Relationship definitions.")

    @model_validator(mode="after")
    def validate_unique_entities(self) -> "ProjectionManifest":
        """
        Validates that all entity names are unique.
        """
        names = [e.name for e in self.entities]
        if len(names) != len(set(names)):
            # Find duplicates
            seen: set[str] = set()
            duplicates = set()
            for x in names:
                if x in seen:
                    duplicates.add(x)
                seen.add(x)
            raise ValueError(f"Duplicate entity names found: {', '.join(duplicates)}")
        return self

    @model_validator(mode="after")
    def validate_relationship_endpoints(self) -> "ProjectionManifest":
        """
        Validates that all relationships connect to entities defined in the manifest.
        """
        entity_names = {e.name for e in self.entities}
        for i, rel in enumerate(self.relationships):
            if rel.start_node not in entity_names:
                raise ValueError(
                    f"Relationship '{rel.name}' (index {i}) has invalid start_node '{rel.start_node}'. "
                    f"Must be one of: {', '.join(sorted(entity_names))}"
                )
            if rel.end_node not in entity_names:
                raise ValueError(
                    f"Relationship '{rel.name}' (index {i}) has invalid end_node '{rel.end_node}'. "
                    f"Must be one of: {', '.join(sorted(entity_names))}"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectionManifest":
        """
        Loads a ProjectionManifest from a YAML file.

        Args:
            path: The path to the YAML file.

        Returns:
            A validated ProjectionManifest instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class GraphJob(BaseModel):
    """
    Represents the runtime state of a Graph Projection Job.

    Attributes:
        id: Unique identifier for the job.
        manifest_path: Path to the manifest file used for this job.
        status: Current status of the job.
        metrics: Performance metrics collected during execution.
    """

    id: UUID = Field(description="Unique identifier for the job.")
    manifest_path: str = Field(description="Path to the manifest file used for this job.")
    status: Literal["RESOLVING", "PROJECTING", "COMPUTING", "COMPLETE", "FAILED"] = Field(
        description="Current status of the job."
    )
    metrics: dict[str, int | float] = Field(
        default_factory=lambda: {
            "nodes_created": 0.0,
            "edges_created": 0.0,
            "ontology_misses": 0.0,
            "ontology_cache_hits": 0.0,
        },
        description="Performance metrics collected during execution.",
    )

    @model_validator(mode="after")
    def validate_metrics_keys(self) -> "GraphJob":
        """
        Validates that the metrics dictionary contains all required keys.
        """
        required_keys = {"nodes_created", "edges_created", "ontology_misses", "ontology_cache_hits"}
        missing_keys = required_keys - self.metrics.keys()
        if missing_keys:
            raise ValueError(f"Missing required metrics keys: {', '.join(sorted(missing_keys))}")
        return self

    @model_validator(mode="after")
    def validate_metrics_non_negative(self) -> "GraphJob":
        """
        Validates that all metric values are non-negative.
        """
        for key, value in self.metrics.items():
            if value < 0:
                raise ValueError(f"Metric '{key}' cannot be negative (got {value})")
        return self


class AnalysisAlgo(str, Enum):
    """
    Enumeration of supported graph analysis algorithms.
    """

    PAGERANK = "pagerank"
    SHORTEST_PATH = "shortest_path"
    LOUVAIN = "louvain"


class GraphAnalysisRequest(BaseModel):
    """
    Request object for running graph analysis algorithms.

    Attributes:
        center_node_id: The ID of the center node for subgraph projection.
        target_node_id: The ID of the target node (required for shortest path).
        algorithm: The algorithm to run.
        depth: The depth of the subgraph projection (K-Hops).
        write_property: The property key to write results back to (for PageRank/Louvain).
    """

    center_node_id: str = Field(description="The ID of the center node for subgraph projection.")
    target_node_id: str | None = Field(
        default=None, description="The ID of the target node (required for shortest path)."
    )
    algorithm: AnalysisAlgo = Field(description="The algorithm to run.")
    depth: int = Field(default=2, ge=1, description="The depth of the subgraph projection (K-Hops).")
    write_property: str = Field(
        default="pagerank_score",
        description="The property key to write results back to (for PageRank/Louvain).",
    )


class LinkPredictionMethod(str, Enum):
    """
    Enumeration of supported link prediction methods.
    """

    HEURISTIC = "heuristic"
    SEMANTIC = "semantic"


class LinkPredictionRequest(BaseModel):
    """
    Request object for running link prediction.

    Attributes:
        method: The method to use for prediction (HEURISTIC or SEMANTIC).
        heuristic_query: The Cypher query to execute for rule-based prediction.
                         Required if method is HEURISTIC.
    """

    method: LinkPredictionMethod = Field(description="The method to use for prediction (HEURISTIC or SEMANTIC).")
    heuristic_query: str | None = Field(
        default=None,
        description="The Cypher query to execute for rule-based prediction. Required if method is HEURISTIC.",
    )
    threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Minimum cosine similarity threshold (for SEMANTIC)."
    )
    embedding_property: str = Field(
        default="embedding", description="Property key for vector embedding (for SEMANTIC)."
    )
    relationship_type: str = Field(
        default="SEMANTIC_LINK", description="Type of relationship to create (for SEMANTIC)."
    )
    source_label: str | None = Field(default=None, description="Label for source nodes (Required for SEMANTIC).")
    target_label: str | None = Field(default=None, description="Label for target nodes (Required for SEMANTIC).")

    @model_validator(mode="after")
    def validate_heuristic_query(self) -> "LinkPredictionRequest":
        """
        Validates that heuristic_query is present and non-empty if method is HEURISTIC.
        """
        if self.method == LinkPredictionMethod.HEURISTIC:
            if not self.heuristic_query or not self.heuristic_query.strip():
                raise ValueError(
                    "heuristic_query is required and cannot be empty/whitespace for HEURISTIC prediction method."
                )
        return self

    @model_validator(mode="after")
    def validate_semantic_params(self) -> "LinkPredictionRequest":
        """
        Validates that source_label and target_label are present if method is SEMANTIC.
        """
        if self.method == LinkPredictionMethod.SEMANTIC:
            if not self.source_label or not self.target_label:
                raise ValueError("source_label and target_label are required for SEMANTIC prediction.")
        return self
