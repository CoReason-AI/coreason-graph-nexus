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
from pathlib import Path
from typing import Any, Generator

import pytest
import yaml
from pydantic import ValidationError

from coreason_graph_nexus.models import (
    Entity,
    GraphJob,
    ProjectionManifest,
    PropertyMapping,
    Relationship,
)


@pytest.fixture(name="valid_manifest_data")
def valid_manifest_data_fixture() -> Generator[dict[str, Any], None, None]:
    data = {
        "version": "1.0",
        "source_connection": "postgres://db",
        "entities": [
            {
                "name": "Drug",
                "source_table": "products",
                "id_column": "id",
                "ontology_mapping": "RxNorm",
                "properties": [{"source": "prod_name", "target": "name"}],
            },
            {
                "name": "AdverseEvent",
                "source_table": "events",
                "id_column": "id",
                "ontology_mapping": "MedDRA",
                "properties": [{"source": "term", "target": "name"}],
            },
        ],
        "relationships": [
            {
                "name": "CAUSED",
                "source_table": "links",
                "start_node": "Drug",
                "start_key": "drug_id",
                "end_node": "AdverseEvent",
                "end_key": "event_id",
            }
        ],
    }
    yield data


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


def test_projection_manifest_creation(valid_manifest_data: dict[str, Any]) -> None:
    manifest = ProjectionManifest(**valid_manifest_data)
    assert manifest.version == "1.0"
    assert len(manifest.entities) == 2
    assert len(manifest.relationships) == 1


def test_projection_manifest_validation_error() -> None:
    with pytest.raises(ValidationError):
        ProjectionManifest(
            version="1.0",
            source_connection="postgres://gold_db",
            entities=[],
            relationships="invalid",  # type: ignore # noqa: PGH003
        )


def test_manifest_validation_invalid_start_node(valid_manifest_data: dict[str, Any]) -> None:
    data = valid_manifest_data
    # Set start_node to something that doesn't exist in entities
    data["relationships"][0]["start_node"] = "NonExistentEntity"

    with pytest.raises(ValidationError) as excinfo:
        ProjectionManifest(**data)

    assert "invalid start_node 'NonExistentEntity'" in str(excinfo.value)
    assert "Must be one of: AdverseEvent, Drug" in str(excinfo.value)


def test_manifest_validation_invalid_end_node(valid_manifest_data: dict[str, Any]) -> None:
    data = valid_manifest_data
    # Set end_node to something that doesn't exist in entities
    data["relationships"][0]["end_node"] = "GhostEntity"

    with pytest.raises(ValidationError) as excinfo:
        ProjectionManifest(**data)

    assert "invalid end_node 'GhostEntity'" in str(excinfo.value)


def test_manifest_duplicate_entity_names(valid_manifest_data: dict[str, Any]) -> None:
    data = valid_manifest_data
    # Duplicate "Drug" entity
    data["entities"].append(data["entities"][0].copy())

    with pytest.raises(ValidationError) as excinfo:
        ProjectionManifest(**data)

    assert "Duplicate entity names found: Drug" in str(excinfo.value)


def test_manifest_self_referencing_relationship(valid_manifest_data: dict[str, Any]) -> None:
    data = valid_manifest_data
    # Add self-loop: Drug -> Drug
    data["relationships"].append(
        {
            "name": "INTERACTS_WITH",
            "source_table": "interactions",
            "start_node": "Drug",
            "start_key": "drug_a",
            "end_node": "Drug",
            "end_key": "drug_b",
        }
    )

    manifest = ProjectionManifest(**data)
    assert len(manifest.relationships) == 2
    assert manifest.relationships[1].start_node == "Drug"
    assert manifest.relationships[1].end_node == "Drug"


def test_manifest_case_sensitivity(valid_manifest_data: dict[str, Any]) -> None:
    data = valid_manifest_data
    # Change relationship to reference "drug" (lowercase) instead of "Drug"
    data["relationships"][0]["start_node"] = "drug"

    with pytest.raises(ValidationError) as excinfo:
        ProjectionManifest(**data)

    # Should fail because "drug" != "Drug"
    assert "invalid start_node 'drug'" in str(excinfo.value)


def test_manifest_empty_strings(valid_manifest_data: dict[str, Any]) -> None:
    data = valid_manifest_data
    # Set an entity name to empty string
    data["entities"][0]["name"] = ""

    with pytest.raises(ValidationError) as excinfo:
        ProjectionManifest(**data)

    assert "String should have at least 1 character" in str(excinfo.value)


def test_manifest_from_yaml(tmp_path: Path, valid_manifest_data: dict[str, Any]) -> None:
    yaml_file = tmp_path / "manifest.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(valid_manifest_data, f)

    manifest = ProjectionManifest.from_yaml(yaml_file)
    assert manifest.version == "1.0"
    assert len(manifest.entities) == 2
    assert manifest.entities[0].name == "Drug"


def test_from_yaml_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        ProjectionManifest.from_yaml("non_existent_file.yaml")


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
            status="INVALID_STATUS",  # type: ignore # noqa: PGH003
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
