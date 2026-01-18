# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient
from coreason_graph_nexus.interfaces import OntologyResolver, SourceAdapter
from coreason_graph_nexus.models import Entity, GraphJob, ProjectionManifest, PropertyMapping, Relationship
from coreason_graph_nexus.projector import ProjectionEngine

# --- Mocks ---


class MockSourceAdapter(SourceAdapter):
    """
    In-memory source adapter for testing.
    """

    def __init__(self, data: dict[str, list[dict[str, Any]]]):
        self.data = data
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_table(self, table_name: str) -> Iterator[dict[str, Any]]:
        if not self.connected:
            raise ConnectionError("Not connected")
        # Yield rows
        yield from self.data.get(table_name, [])


class MockOntologyResolver(OntologyResolver):
    """
    Mock resolver that returns predictable IDs.
    """

    def __init__(self, mappings: dict[str, str] | None = None):
        self.mappings = mappings or {}

    def resolve(self, term: str) -> str | None:
        return self.mappings.get(term)


# --- Fixtures ---


@pytest.fixture
def mock_neo4j_client() -> MagicMock:
    return MagicMock(spec=Neo4jClient)


@pytest.fixture
def mock_adapter() -> MockSourceAdapter:
    data = {
        "drugs": [
            {"drug_id": "1001", "name": "Aspirin", "type": "Analgesic"},
            {"drug_id": "1002", "name": "Tylenol", "type": "Analgesic"},
        ],
        "diseases": [
            {"code": "D01", "desc": "Headache"},
        ],
        "treatments": [
            {"drug_ref": "1001", "disease_ref": "D01", "evidence": "Strong"},
            {"drug_ref": "1002", "disease_ref": "D01", "evidence": "Medium"},
        ],
        "empty_table": [],
    }
    adapter = MockSourceAdapter(data)
    adapter.connect()
    return adapter


@pytest.fixture
def mock_resolver() -> MockOntologyResolver:
    return MockOntologyResolver({"1001": "RxNorm:111", "1002": "RxNorm:222", "D01": "SNOMED:333"})


@pytest.fixture
def drug_entity() -> Entity:
    return Entity(
        name="Drug",
        source_table="drugs",
        id_column="drug_id",
        ontology_mapping="RxNorm",
        properties=[
            PropertyMapping(source="name", target="label"),
            PropertyMapping(source="type", target="category"),
        ],
    )


@pytest.fixture
def drug_only_manifest(drug_entity: Entity) -> ProjectionManifest:
    return ProjectionManifest(
        version="1.0",
        source_connection="mock://db",
        entities=[drug_entity],
        relationships=[],
    )


@pytest.fixture
def sample_manifest(drug_entity: Entity) -> ProjectionManifest:
    disease_entity = Entity(
        name="Disease",
        source_table="diseases",
        id_column="code",
        ontology_mapping="SNOMED",
        properties=[
            PropertyMapping(source="desc", target="label"),
        ],
    )
    return ProjectionManifest(
        version="1.0",
        source_connection="mock://db",
        entities=[drug_entity, disease_entity],
        relationships=[
            Relationship(
                name="TREATS",
                source_table="treatments",
                start_node="Drug",
                start_key="drug_ref",
                end_node="Disease",
                end_key="disease_ref",
            )
        ],
    )


@pytest.fixture
def graph_job() -> GraphJob:
    return GraphJob(id=uuid4(), manifest_path="test.yaml", status="RESOLVING")


# --- Tests ---


def test_ingest_entities_happy_path(
    mock_neo4j_client: MagicMock,
    mock_adapter: MockSourceAdapter,
    mock_resolver: MockOntologyResolver,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    engine = ProjectionEngine(mock_neo4j_client, mock_resolver)

    engine.ingest_entities(drug_only_manifest, mock_adapter, graph_job, batch_size=10)

    # Verify status update
    assert graph_job.status == "PROJECTING"

    # Verify metrics
    assert graph_job.metrics["nodes_created"] == 2.0
    assert graph_job.metrics["ontology_misses"] == 0.0

    # Verify Neo4j call
    mock_neo4j_client.merge_nodes.assert_called_once()
    args, kwargs = mock_neo4j_client.merge_nodes.call_args
    label = args[0]
    data = args[1]

    assert label == "Drug"
    assert len(data) == 2

    # Check data transformation
    # First item: ID resolved to RxNorm:111
    assert data[0]["id"] == "RxNorm:111"
    assert data[0]["label"] == "Aspirin"
    assert data[0]["category"] == "Analgesic"


def test_ingest_relationships_happy_path(
    mock_neo4j_client: MagicMock,
    mock_adapter: MockSourceAdapter,
    mock_resolver: MockOntologyResolver,
    sample_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    engine = ProjectionEngine(mock_neo4j_client, mock_resolver)

    engine.ingest_relationships(sample_manifest, mock_adapter, graph_job, batch_size=10)

    # Verify status
    assert graph_job.status == "PROJECTING"

    # Verify metrics
    assert graph_job.metrics["edges_created"] == 2.0
    assert graph_job.metrics["ontology_misses"] == 0.0

    # Verify Neo4j call
    mock_neo4j_client.merge_relationships.assert_called_once()
    kwargs = mock_neo4j_client.merge_relationships.call_args[1]

    assert kwargs["start_label"] == "Drug"
    assert kwargs["end_label"] == "Disease"
    assert kwargs["rel_type"] == "TREATS"
    assert kwargs["start_data_key"] == "drug_ref"
    assert kwargs["end_data_key"] == "disease_ref"

    data = kwargs["data"]
    assert len(data) == 2

    # Check ID resolution
    # 1001 -> RxNorm:111, D01 -> SNOMED:333
    assert data[0]["drug_ref"] == "RxNorm:111"
    assert data[0]["disease_ref"] == "SNOMED:333"
    assert data[0]["evidence"] == "Strong"


def test_ingest_relationships_ontology_miss(
    mock_neo4j_client: MagicMock,
    mock_adapter: MockSourceAdapter,
    sample_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    # Empty resolver -> miss
    resolver = MockOntologyResolver({})
    engine = ProjectionEngine(mock_neo4j_client, resolver)

    engine.ingest_relationships(sample_manifest, mock_adapter, graph_job)

    assert graph_job.metrics["edges_created"] == 2.0
    # 2 rows * 2 lookups (start/end) = 4 misses
    assert graph_job.metrics["ontology_misses"] == 4.0

    # Verify fallback
    kwargs = mock_neo4j_client.merge_relationships.call_args[1]
    data = kwargs["data"]

    assert data[0]["drug_ref"] == "1001"
    assert data[0]["disease_ref"] == "D01"


def test_ingest_relationships_missing_keys(
    mock_neo4j_client: MagicMock,
    sample_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    data: dict[str, list[dict[str, Any]]] = {
        "treatments": [
            {"drug_ref": "1001", "disease_ref": "D01"},
            {"evidence": "Bad Row"},  # Missing keys
            {"drug_ref": None, "disease_ref": "D01"},  # None key
        ]
    }
    adapter = MockSourceAdapter(data)
    adapter.connect()

    engine = ProjectionEngine(mock_neo4j_client, MockOntologyResolver())
    engine.ingest_relationships(sample_manifest, adapter, graph_job)

    assert graph_job.metrics["edges_created"] == 1.0

    kwargs = mock_neo4j_client.merge_relationships.call_args[1]
    assert len(kwargs["data"]) == 1


def test_ingest_entities_batching(
    mock_neo4j_client: MagicMock,
    mock_resolver: MockOntologyResolver,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    # Generate 25 items
    data = {"drugs": [{"drug_id": str(i), "name": f"Drug{i}"} for i in range(25)]}
    adapter = MockSourceAdapter(data)
    adapter.connect()

    engine = ProjectionEngine(mock_neo4j_client, mock_resolver)

    # Set batch size to 10
    engine.ingest_entities(drug_only_manifest, adapter, graph_job, batch_size=10)

    # Should be called 3 times (10, 10, 5)
    assert mock_neo4j_client.merge_nodes.call_count == 3
    assert graph_job.metrics["nodes_created"] == 25.0


def test_ingest_entities_missing_id(
    mock_neo4j_client: MagicMock,
    mock_resolver: MockOntologyResolver,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    data: dict[str, list[dict[str, Any]]] = {
        "drugs": [
            {"drug_id": "1", "name": "Valid"},
            {"name": "Invalid"},  # Missing ID column
            {"drug_id": None, "name": "NoneID"},
        ]
    }
    adapter = MockSourceAdapter(data)
    adapter.connect()

    engine = ProjectionEngine(mock_neo4j_client, mock_resolver)
    engine.ingest_entities(drug_only_manifest, adapter, graph_job)

    # Should only ingest 1
    assert graph_job.metrics["nodes_created"] == 1.0
    args, _ = mock_neo4j_client.merge_nodes.call_args
    assert len(args[1]) == 1


def test_ingest_entities_ontology_miss(
    mock_neo4j_client: MagicMock,
    mock_adapter: MockSourceAdapter,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    # Empty resolver -> all miss
    resolver = MockOntologyResolver({})
    engine = ProjectionEngine(mock_neo4j_client, resolver)

    engine.ingest_entities(drug_only_manifest, mock_adapter, graph_job)

    assert graph_job.metrics["nodes_created"] == 2.0
    assert graph_job.metrics["ontology_misses"] == 2.0

    # Verify fallback to source ID
    args, _ = mock_neo4j_client.merge_nodes.call_args
    data = args[1]
    assert data[0]["id"] == "1001"  # Fallback


def test_ingest_entities_empty_table(
    mock_neo4j_client: MagicMock,
    mock_adapter: MockSourceAdapter,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    # Point manifest to empty table
    drug_only_manifest.entities[0].source_table = "empty_table"

    engine = ProjectionEngine(mock_neo4j_client, MockOntologyResolver())
    engine.ingest_entities(drug_only_manifest, mock_adapter, graph_job)

    mock_neo4j_client.merge_nodes.assert_not_called()
    assert graph_job.metrics["nodes_created"] == 0.0


def test_ingest_entities_exception_handling(
    mock_neo4j_client: MagicMock,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    # Adapter raises error
    adapter = MagicMock(spec=SourceAdapter)
    adapter.read_table.side_effect = Exception("DB Error")

    engine = ProjectionEngine(mock_neo4j_client, MockOntologyResolver())

    with pytest.raises(Exception, match="DB Error"):
        engine.ingest_entities(drug_only_manifest, adapter, graph_job)


def test_ingest_entities_property_mapping(
    mock_neo4j_client: MagicMock,
    mock_adapter: MockSourceAdapter,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    # Test that only mapped properties are included
    engine = ProjectionEngine(mock_neo4j_client, MockOntologyResolver())
    engine.ingest_entities(drug_only_manifest, mock_adapter, graph_job)

    args, _ = mock_neo4j_client.merge_nodes.call_args
    item = args[1][0]

    # 'type' is in source but mapped to 'category'
    assert "type" not in item
    assert "category" in item
    # 'name' mapped to 'label'
    assert "name" not in item
    assert "label" in item


# --- Complex Scenarios ---


def test_ingest_entities_many_to_one_resolution(
    mock_neo4j_client: MagicMock,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    """Test merging of multiple source entities into a single resolved graph node."""
    # Data: Two different source items that map to the same concept
    # "Tyl" -> RxNorm:111, "Para" -> RxNorm:111
    data: dict[str, list[dict[str, Any]]] = {
        "drugs": [
            {"drug_id": "Tyl", "name": "Tylenol", "type": "Brand"},
            {"drug_id": "Para", "name": "Paracetamol", "type": "Generic"},
        ]
    }
    adapter = MockSourceAdapter(data)
    adapter.connect()

    # Resolver maps both to same ID
    resolver = MockOntologyResolver({"Tyl": "RxNorm:111", "Para": "RxNorm:111"})

    engine = ProjectionEngine(mock_neo4j_client, resolver)
    engine.ingest_entities(drug_only_manifest, adapter, graph_job)

    # 1. Verify metrics: 2 source records processed (nodes_created metric tracks rows processed essentially)
    assert graph_job.metrics["nodes_created"] == 2.0

    # 2. Verify Neo4j data
    args, _ = mock_neo4j_client.merge_nodes.call_args
    batch_data = args[1]
    assert len(batch_data) == 2

    # Both should have same 'id'
    assert batch_data[0]["id"] == "RxNorm:111"
    assert batch_data[1]["id"] == "RxNorm:111"

    # Properties should differ
    assert batch_data[0]["label"] == "Tylenol"
    assert batch_data[1]["label"] == "Paracetamol"


def test_ingest_entities_duplicate_updates(
    mock_neo4j_client: MagicMock,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    """Test duplicate rows for same ID in the stream."""
    data: dict[str, list[dict[str, Any]]] = {
        "drugs": [
            {"drug_id": "1", "name": "V1", "type": "A"},
            {"drug_id": "1", "name": "V2", "type": "B"},
        ]
    }
    adapter = MockSourceAdapter(data)
    adapter.connect()

    # Resolver behaves as identity if not found
    resolver = MockOntologyResolver({})

    engine = ProjectionEngine(mock_neo4j_client, resolver)
    engine.ingest_entities(drug_only_manifest, adapter, graph_job)

    args, _ = mock_neo4j_client.merge_nodes.call_args
    batch_data = args[1]

    # Should contain both records with same ID
    assert len(batch_data) == 2
    assert batch_data[0]["id"] == "1"
    assert batch_data[0]["label"] == "V1"
    assert batch_data[1]["id"] == "1"
    assert batch_data[1]["label"] == "V2"


def test_ingest_entities_null_properties(
    mock_neo4j_client: MagicMock,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    """Test handling of None/Null values in properties."""
    data: dict[str, list[dict[str, Any]]] = {
        "drugs": [
            {"drug_id": "1", "name": "Valid", "type": None},
        ]
    }
    adapter = MockSourceAdapter(data)
    adapter.connect()

    engine = ProjectionEngine(mock_neo4j_client, MockOntologyResolver())
    engine.ingest_entities(drug_only_manifest, adapter, graph_job)

    args, _ = mock_neo4j_client.merge_nodes.call_args
    batch_data = args[1]

    # Verify None is propagated to the dictionary
    assert batch_data[0]["category"] is None
    assert batch_data[0]["label"] == "Valid"


def test_ingest_entities_special_characters(
    mock_neo4j_client: MagicMock,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    """Test strings with special characters."""
    special_name = "Drug \"with\" 'quotes' & \n newlines \U0001f600"
    data: dict[str, list[dict[str, Any]]] = {
        "drugs": [
            {"drug_id": "special", "name": special_name, "type": "Test"},
        ]
    }
    adapter = MockSourceAdapter(data)
    adapter.connect()

    engine = ProjectionEngine(mock_neo4j_client, MockOntologyResolver())
    engine.ingest_entities(drug_only_manifest, adapter, graph_job)

    args, _ = mock_neo4j_client.merge_nodes.call_args
    batch_data = args[1]

    assert batch_data[0]["label"] == special_name


def test_ingest_entities_list_properties(
    mock_neo4j_client: MagicMock,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    """Test properties that are lists (e.g., tags)."""
    data: dict[str, list[dict[str, Any]]] = {
        "drugs": [
            {"drug_id": "1", "name": ["Alias1", "Alias2"], "type": "Test"},
        ]
    }
    adapter = MockSourceAdapter(data)
    adapter.connect()

    engine = ProjectionEngine(mock_neo4j_client, MockOntologyResolver())
    engine.ingest_entities(drug_only_manifest, adapter, graph_job)

    args, _ = mock_neo4j_client.merge_nodes.call_args
    batch_data = args[1]

    assert batch_data[0]["label"] == ["Alias1", "Alias2"]


def test_ingest_relationships_mixed_resolution_states(
    mock_neo4j_client: MagicMock,
    sample_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    """
    Test mixed resolution success/failure and self-loops.

    Data:
      - R1: Known -> Unknown (Start resolves, End misses)
      - R2: Unknown -> Known (Start misses, End resolves)
      - R3: Known -> Known (Start resolves, End resolves -> Self Loop if same ID)
    """
    data = {
        "treatments": [
            {"drug_ref": "Known", "disease_ref": "Unknown"},
            {"drug_ref": "Unknown", "disease_ref": "Known"},
            {"drug_ref": "Known", "disease_ref": "Known"},
        ]
    }
    adapter = MockSourceAdapter(data)
    adapter.connect()

    # Resolver: "Known" -> "ID1"
    resolver = MockOntologyResolver({"Known": "ID1"})

    engine = ProjectionEngine(mock_neo4j_client, resolver)
    engine.ingest_relationships(sample_manifest, adapter, graph_job)

    # Metrics
    assert graph_job.metrics["edges_created"] == 3.0
    # R1 misses end, R2 misses start, R3 misses none. Total 2.
    assert graph_job.metrics["ontology_misses"] == 2.0

    kwargs = mock_neo4j_client.merge_relationships.call_args[1]
    batch_data = kwargs["data"]

    # R1: ID1 -> Unknown
    assert batch_data[0]["drug_ref"] == "ID1"
    assert batch_data[0]["disease_ref"] == "Unknown"

    # R2: Unknown -> ID1
    assert batch_data[1]["drug_ref"] == "Unknown"
    assert batch_data[1]["disease_ref"] == "ID1"

    # R3: ID1 -> ID1 (Self loop)
    assert batch_data[2]["drug_ref"] == "ID1"
    assert batch_data[2]["disease_ref"] == "ID1"


def test_ingest_relationships_empty_keys_and_type_coercion(
    mock_neo4j_client: MagicMock,
    sample_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    """Test skipping of empty/None keys and handling of integer keys."""
    data: dict[str, list[dict[str, Any]]] = {
        "treatments": [
            {"drug_ref": 123, "disease_ref": "Valid"},  # Int key
            {"drug_ref": "", "disease_ref": "Valid"},  # Empty start -> Skip
            {"drug_ref": "Valid", "disease_ref": None},  # None end -> Skip
            {"drug_ref": "Valid", "disease_ref": ""},  # Empty end -> Skip
        ]
    }
    adapter = MockSourceAdapter(data)
    adapter.connect()

    # Resolver: "123" -> "ID123", "Valid" -> "IDValid"
    resolver = MockOntologyResolver({"123": "ID123", "Valid": "IDValid"})

    engine = ProjectionEngine(mock_neo4j_client, resolver)
    engine.ingest_relationships(sample_manifest, adapter, graph_job)

    # Should only process the first one
    assert graph_job.metrics["edges_created"] == 1.0

    kwargs = mock_neo4j_client.merge_relationships.call_args[1]
    batch_data = kwargs["data"]

    assert len(batch_data) == 1
    assert batch_data[0]["drug_ref"] == "ID123"
    assert batch_data[0]["disease_ref"] == "IDValid"


def test_ingest_relationships_batching(
    mock_neo4j_client: MagicMock,
    mock_resolver: MockOntologyResolver,
    sample_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    """Verify batching for relationships."""
    # 25 rows
    data = {"treatments": [{"drug_ref": str(i), "disease_ref": f"D{i}"} for i in range(25)]}
    adapter = MockSourceAdapter(data)
    adapter.connect()

    engine = ProjectionEngine(mock_neo4j_client, mock_resolver)
    engine.ingest_relationships(sample_manifest, adapter, graph_job, batch_size=10)

    # 3 batches (10, 10, 5)
    assert mock_neo4j_client.merge_relationships.call_count == 3
    assert graph_job.metrics["edges_created"] == 25.0


def test_ingest_relationships_residual_flush(
    mock_neo4j_client: MagicMock,
    mock_resolver: MockOntologyResolver,
    sample_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    """Verify that a small batch that doesn't trigger the loop flush is flushed at the end."""
    # 3 rows, batch size 10
    data = {"treatments": [{"drug_ref": str(i), "disease_ref": f"D{i}"} for i in range(3)]}
    adapter = MockSourceAdapter(data)
    adapter.connect()

    engine = ProjectionEngine(mock_neo4j_client, mock_resolver)
    engine.ingest_relationships(sample_manifest, adapter, graph_job, batch_size=10)

    # Should be called once (at end)
    assert mock_neo4j_client.merge_relationships.call_count == 1
    assert graph_job.metrics["edges_created"] == 3.0


def test_ingest_relationships_exception_handling(
    mock_neo4j_client: MagicMock,
    sample_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    """Verify exception handling during relationship ingestion."""
    # Adapter raises error
    adapter = MagicMock(spec=SourceAdapter)
    adapter.read_table.side_effect = Exception("Source DB Error")

    engine = ProjectionEngine(mock_neo4j_client, MockOntologyResolver())

    with pytest.raises(Exception, match="Source DB Error"):
        engine.ingest_relationships(sample_manifest, adapter, graph_job)


def test_ingest_entities_mixed_scenario(
    mock_neo4j_client: MagicMock,
    drug_only_manifest: ProjectionManifest,
    graph_job: GraphJob,
) -> None:
    """
    Test a mix of:
    - Valid row
    - Missing ID row (skip)
    - Ontology miss row (process but fallback)
    """
    data = {
        "drugs": [
            {"drug_id": "Valid1", "name": "V1"},
            {"name": "MissingID"},
            {"drug_id": "Miss1", "name": "M1"},
        ]
    }
    adapter = MockSourceAdapter(data)
    adapter.connect()

    resolver = MockOntologyResolver({"Valid1": "Resolved1"})

    engine = ProjectionEngine(mock_neo4j_client, resolver)
    engine.ingest_entities(drug_only_manifest, adapter, graph_job, batch_size=10)

    # 2 rows processed (Valid1, Miss1). MissingID is skipped.
    assert graph_job.metrics["nodes_created"] == 2.0

    # 1 ontology miss (Miss1)
    assert graph_job.metrics["ontology_misses"] == 1.0

    args, _ = mock_neo4j_client.merge_nodes.call_args
    batch_data = args[1]

    assert len(batch_data) == 2
    assert batch_data[0]["id"] == "Resolved1"
    assert batch_data[1]["id"] == "Miss1"
