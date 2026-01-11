# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class PropertyMapping(BaseModel):
    source: str
    target: str


class Entity(BaseModel):
    name: str
    source_table: str
    id_column: str
    ontology_mapping: str
    properties: list[PropertyMapping]


class Relationship(BaseModel):
    name: str
    source_table: str
    start_node: str
    start_key: str
    end_node: str
    end_key: str


class ProjectionManifest(BaseModel):
    version: str
    source_connection: str
    entities: list[Entity]
    relationships: list[Relationship]


class GraphJob(BaseModel):
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
