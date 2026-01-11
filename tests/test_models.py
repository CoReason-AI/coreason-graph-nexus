# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

import uuid

import pytest
from pydantic import ValidationError

from coreason_graph_nexus.models import (
    Entity,
    GraphJob,
    ProjectionManifest,
    PropertyMapping,
    Relationship,
)


def test_property_mapping_creation() -> None:
    mapping = PropertyMapping(source="product_name", target="name")
    assert mapping.source == "product_name"
    assert mapping.target == "name"


def test_entity_creation() -> None:
    mapping = PropertyMapping(source="product_name", target="name")
    entity = Entity(
        name="Drug",
        source_table="dim_products",
        id_column="product_code",
        ontology_mapping="RxNorm",
        properties=[mapping],
    )
    assert entity.name == "Drug"
    assert len(entity.properties) == 1


def test_relationship_creation() -> None:
    rel = Relationship(
        name="REPORTED_EVENT",
        source_table="fact_adverse_events",
        start_node="Drug",
        start_key="product_code",
        end_node="AdverseEvent",
        end_key="event_term",
    )
    assert rel.name == "REPORTED_EVENT"
    assert rel.start_node == "Drug"
    assert rel.end_node == "AdverseEvent"


def test_projection_manifest_creation() -> None:
    mapping = PropertyMapping(source="product_name", target="name")
    entity = Entity(
        name="Drug",
        source_table="dim_products",
        id_column="product_code",
        ontology_mapping="RxNorm",
        properties=[mapping],
    )
    rel = Relationship(
        name="REPORTED_EVENT",
        source_table="fact_adverse_events",
        start_node="Drug",
        start_key="product_code",
        end_node="AdverseEvent",
        end_key="event_term",
    )
    manifest = ProjectionManifest(
        version="1.0",
        source_connection="postgres://gold_db",
        entities=[entity],
        relationships=[rel],
    )
    assert manifest.version == "1.0"
    assert len(manifest.entities) == 1
    assert len(manifest.relationships) == 1


def test_projection_manifest_validation_error() -> None:
    with pytest.raises(ValidationError):
        ProjectionManifest(
            version="1.0",
            source_connection="postgres://gold_db",
            entities=[],
            relationships="invalid",  # type: ignore
        )


def test_graph_job_creation() -> None:
    job_id = uuid.uuid4()
    job = GraphJob(
        id=job_id,
        manifest_path="/path/to/manifest.yaml",
        status="RESOLVING",
    )
    assert job.id == job_id
    assert job.status == "RESOLVING"
    assert job.metrics["nodes_created"] == 0


def test_graph_job_validation_error() -> None:
    with pytest.raises(ValidationError):
        GraphJob(
            id=uuid.uuid4(),
            manifest_path="/path/to/manifest.yaml",
            status="INVALID_STATUS",  # type: ignore
        )


def test_graph_job_metrics_default() -> None:
    job = GraphJob(
        id=uuid.uuid4(),
        manifest_path="/path/to/manifest.yaml",
        status="COMPLETE",
    )
    assert job.metrics == {
        "nodes_created": 0,
        "edges_created": 0,
        "ontology_misses": 0,
    }


def test_graph_job_metrics_custom() -> None:
    metrics: dict[str, int | float] = {
        "nodes_created": 100.0,
        "edges_created": 200.0,
        "ontology_misses": 5.0,
    }
    job = GraphJob(
        id=uuid.uuid4(),
        manifest_path="/path/to/manifest.yaml",
        status="COMPLETE",
        metrics=metrics,
    )
    assert job.metrics == metrics
