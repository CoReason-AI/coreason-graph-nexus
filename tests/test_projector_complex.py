# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from typing import Any
import pytest
from uuid import uuid4
from coreason_graph_nexus.projector import ProjectionEngine
from coreason_graph_nexus.models import GraphJob, ProjectionManifest, Entity, Relationship, PropertyMapping

class MockOntologyResolver:
    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping = mapping or {}

    def resolve(self, term: str) -> str | None:
        return self.mapping.get(term)

class MockSourceAdapter:
    def __init__(self, data: dict[str, list[dict[str, Any]]]):
        self.data = data

    def read_table(self, table_name: str) -> Any:
        return iter(self.data.get(table_name, []))

def test_ontology_misses(mocker: Any) -> None:
    """
    Test that when the ontology resolver returns None, the system:
    1. Falls back to the original source ID.
    2. Increments the 'ontology_misses' metric.
    """
    mock_client = mocker.Mock()

    # Resolver knows "known_term" -> "canonical_id", but doesn't know "unknown_term"
    resolver = MockOntologyResolver(mapping={"known_term": "canonical_id"})

    engine = ProjectionEngine(client=mock_client, resolver=resolver) # type: ignore

    # Define Manifest
    entity = Entity(
        name="TestEntity",
        source_table="source_table",
        id_column="id_col",
        ontology_mapping="RxNorm", # Added missing field
        properties=[]
    )
    manifest = ProjectionManifest(
        version="1.0", # Added missing field
        source_connection="sqlite:///", # Added missing field
        entities=[entity],
        relationships=[]
    )

    # Define Data
    # Row 1: Known -> Should use "canonical_id", no miss.
    # Row 2: Unknown -> Should use "unknown_term", +1 miss.
    data = {
        "source_table": [
            {"id_col": "known_term"},
            {"id_col": "unknown_term"}
        ]
    }
    adapter = MockSourceAdapter(data)

    job = GraphJob(id=uuid4(), status="PROJECTING", manifest_path="dummy.yaml") # Fixed fields

    engine.ingest_entities(manifest, adapter, job)

    # Verify Metrics
    # We expect 1 ontology miss (for "unknown_term")
    assert job.metrics.get("ontology_misses") == 1.0
    assert job.metrics.get("nodes_created") == 2.0

    # Verify Calls
    # We expect merge_nodes to be called.
    # The data passed should show the fallback.
    assert mock_client.merge_nodes.call_count > 0
    # Inspect the data passed to merge_nodes
    call_args = mock_client.merge_nodes.call_args
    passed_data = call_args[1]['data'] if 'data' in call_args[1] else call_args[0][1]

    # Item 1: known_term resolved to canonical_id
    assert passed_data[0]['id'] == "canonical_id"
    # Item 2: unknown_term fallback
    assert passed_data[1]['id'] == "unknown_term"

def test_relationship_ontology_misses(mocker: Any) -> None:
    """
    Test ontology misses during relationship ingestion.
    """
    mock_client = mocker.Mock()
    resolver = MockOntologyResolver(mapping={"s1": "c1", "t1": "c2"})
    engine = ProjectionEngine(client=mock_client, resolver=resolver) # type: ignore

    # Entities needed for validation
    eA = Entity(name="A", source_table="t", id_column="i", ontology_mapping="x", properties=[])
    eB = Entity(name="B", source_table="t", id_column="i", ontology_mapping="x", properties=[])

    rel = Relationship(
        name="REL",
        start_node="A",
        end_node="B",
        source_table="rel_table",
        start_key="start_col",
        end_key="end_col"
    )
    manifest = ProjectionManifest(
        version="1.0",
        source_connection="s",
        entities=[eA, eB],
        relationships=[rel]
    )

    # Row 1: s1->t1 (Both resolved) -> c1->c2. Misses: 0
    # Row 2: s1->t_unknown (Target miss) -> c1->t_unknown. Misses: 1
    # Row 3: s_unknown->t1 (Source miss) -> s_unknown->c2. Misses: 1
    # Total misses: 2
    data = {
        "rel_table": [
            {"start_col": "s1", "end_col": "t1"},
            {"start_col": "s1", "end_col": "t_unknown"},
            {"start_col": "s_unknown", "end_col": "t1"}
        ]
    }
    adapter = MockSourceAdapter(data)
    job = GraphJob(id=uuid4(), status="PROJECTING", manifest_path="d.yaml")

    engine.ingest_relationships(manifest, adapter, job)

    assert job.metrics.get("ontology_misses") == 2.0
    assert job.metrics.get("edges_created") == 3.0

    # Verify data passed to merge_relationships
    call_args = mock_client.merge_relationships.call_args
    passed_data = call_args[1]['data'] if 'data' in call_args[1] else call_args[0][5] # 6th arg is data

    # Row 1
    assert passed_data[0]["start_col"] == "c1"
    assert passed_data[0]["end_col"] == "c2"
    # Row 2
    assert passed_data[1]["start_col"] == "c1"
    assert passed_data[1]["end_col"] == "t_unknown"


def test_data_integrity_skipping(mocker: Any) -> None:
    """
    Test that rows with missing mandatory keys are skipped and logged.
    """
    mock_client = mocker.Mock()
    resolver = MockOntologyResolver()
    engine = ProjectionEngine(client=mock_client, resolver=resolver) # type: ignore

    entity = Entity(
        name="E",
        source_table="table",
        id_column="id",
        ontology_mapping="x",
        properties=[]
    )
    manifest = ProjectionManifest(
        version="1.0",
        source_connection="s",
        entities=[entity],
        relationships=[]
    )

    # Row 1: Valid
    # Row 2: Missing ID (None) -> Skip
    # Row 3: Empty ID ("") -> Skip
    data = {
        "table": [
            {"id": "valid"},
            {"id": None},
            {"id": ""},
            {"id": "valid2"}
        ]
    }
    adapter = MockSourceAdapter(data)
    job = GraphJob(id=uuid4(), status="PROJECTING", manifest_path="d.yaml")

    engine.ingest_entities(manifest, adapter, job)

    assert job.metrics.get("nodes_created") == 2.0

    # Verify only 2 items passed
    call_args = mock_client.merge_nodes.call_args
    passed_data = call_args[1]['data'] if 'data' in call_args[1] else call_args[0][1]
    assert len(passed_data) == 2
    assert passed_data[0]['id'] == "valid"
    assert passed_data[1]['id'] == "valid2"


def test_projection_idempotency_verification(mocker: Any) -> None:
    """
    Verify that the projection engine calls the idempotent client methods.
    """
    mock_client = mocker.Mock()
    resolver = MockOntologyResolver()
    engine = ProjectionEngine(client=mock_client, resolver=resolver) # type: ignore

    entity = Entity(name="E", source_table="t", id_column="id", ontology_mapping="x", properties=[])
    manifest = ProjectionManifest(
        version="1.0",
        source_connection="s",
        entities=[entity],
        relationships=[]
    )
    data = {"t": [{"id": "1"}]}
    adapter = MockSourceAdapter(data)
    job = GraphJob(id=uuid4(), status="PROJECTING", manifest_path="d.yaml")

    # Run 1
    engine.ingest_entities(manifest, adapter, job)
    mock_client.merge_nodes.assert_called_once()

    # Run 2
    engine.ingest_entities(manifest, adapter, job)
    assert mock_client.merge_nodes.call_count == 2
