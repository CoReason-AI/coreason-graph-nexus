# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

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

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


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

    name: str = Field(min_length=1)
    source_table: str = Field(min_length=1)
    id_column: str = Field(min_length=1)
    ontology_mapping: str = Field(min_length=1)
    properties: list[PropertyMapping]


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

    name: str = Field(min_length=1)
    source_table: str = Field(min_length=1)
    start_node: str = Field(min_length=1)
    start_key: str = Field(min_length=1)
    end_node: str = Field(min_length=1)
    end_key: str = Field(min_length=1)


class ProjectionManifest(BaseModel):
    """
    The master configuration for a Graph Projection job.

    Attributes:
        version: The version of the manifest schema.
        source_connection: The connection string for the source database.
        entities: A list of Entity definitions.
        relationships: A list of Relationship definitions.
    """

    version: str = Field(min_length=1)
    source_connection: str = Field(min_length=1)
    entities: list[Entity]
    relationships: list[Relationship]

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

    id: UUID
    manifest_path: str
    status: Literal["RESOLVING", "PROJECTING", "COMPUTING", "COMPLETE"]
    metrics: dict[str, int | float] = Field(
        default_factory=lambda: {
            "nodes_created": 0.0,
            "edges_created": 0.0,
            "ontology_misses": 0.0,
        }
    )
