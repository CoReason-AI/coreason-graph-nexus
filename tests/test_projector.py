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
from typing import Any

import pytest
from pytest_mock import MockerFixture

from coreason_graph_nexus.models import Entity, GraphJob, ProjectionManifest, PropertyMapping, Relationship
from coreason_graph_nexus.projector import ProjectionEngine
from tests.test_utils_shared import MockOntologyResolver, MockSourceAdapter


@pytest.fixture
def manifest() -> ProjectionManifest:
    return ProjectionManifest(
        version="1.0",
        source_connection="mock://",
        entities=[
            Entity(
                name="Person",
                source_table="persons",
                id_column="id",
                ontology_mapping="none",
                properties=[PropertyMapping(source="name", target="name")],
            )
        ],
        relationships=[
            Relationship(
                name="KNOWS",
                source_table="knows",
                start_node="Person",
                start_key="p1",
                end_node="Person",
                end_key="p2",
            )
        ],
    )


@pytest.fixture
def adapter() -> MockSourceAdapter:
    data: dict[str, list[dict[str, Any]]] = {
        "persons": [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}],
        "knows": [{"p1": "1", "p2": "2"}],
    }
    return MockSourceAdapter(data)


@pytest.fixture
def resolver() -> MockOntologyResolver:
    return MockOntologyResolver({"1": "Person:1", "2": "Person:2"})


@pytest.mark.asyncio
async def test_ingest_entities(
    mocker: MockerFixture, manifest: ProjectionManifest, adapter: MockSourceAdapter, resolver: MockOntologyResolver
) -> None:
    mock_client = mocker.Mock()
    # Ensure merge_nodes is awaitable
    mock_client.merge_nodes = mocker.AsyncMock()

    engine = ProjectionEngine(client=mock_client, resolver=resolver)
    job = GraphJob(id=uuid.uuid4(), manifest_path="test", status="RESOLVING")

    async with adapter:
        await engine.ingest_entities(manifest, adapter, job)

    assert job.metrics["nodes_created"] == 2.0
    mock_client.merge_nodes.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_relationships(
    mocker: MockerFixture, manifest: ProjectionManifest, adapter: MockSourceAdapter, resolver: MockOntologyResolver
) -> None:
    mock_client = mocker.Mock()
    mock_client.merge_relationships = mocker.AsyncMock()

    engine = ProjectionEngine(client=mock_client, resolver=resolver)
    job = GraphJob(id=uuid.uuid4(), manifest_path="test", status="RESOLVING")

    async with adapter:
        await engine.ingest_relationships(manifest, adapter, job)

    assert job.metrics["edges_created"] == 1.0
    mock_client.merge_relationships.assert_awaited_once()


@pytest.mark.asyncio
async def test_entity_ingestion_with_missing_id(
    mocker: MockerFixture, adapter: MockSourceAdapter, resolver: MockOntologyResolver
) -> None:
    # Setup bad data
    adapter.data["persons"].append({"id": None, "name": "NoID"})

    manifest = ProjectionManifest(
        version="1.0",
        source_connection="mock://",
        entities=[
            Entity(
                name="Person",
                source_table="persons",
                id_column="id",
                ontology_mapping="none",
                properties=[PropertyMapping(source="name", target="name")],
            )
        ],
        relationships=[],
    )

    mock_client = mocker.Mock()
    mock_client.merge_nodes = mocker.AsyncMock()

    engine = ProjectionEngine(client=mock_client, resolver=resolver)
    job = GraphJob(id=uuid.uuid4(), manifest_path="test", status="RESOLVING")

    async with adapter:
        await engine.ingest_entities(manifest, adapter, job)

    # 3 items in adapter, but 1 has no ID, so only 2 should be processed in the batch
    # wait, the batching logic filters None results.
    # process_entity_row returns None if source_id is None.
    # So consumer receives batch of size 2.
    # However, I added bad data to existing 2, so total 3 rows, 1 skipped -> 2 valid.

    assert job.metrics["nodes_created"] == 2.0


@pytest.mark.asyncio
async def test_relationship_ingestion_with_missing_keys(
    mocker: MockerFixture, adapter: MockSourceAdapter, resolver: MockOntologyResolver
) -> None:
    adapter.data["knows"].append({"p1": "1", "p2": None})

    manifest = ProjectionManifest(
        version="1.0",
        source_connection="mock://",
        entities=[
            Entity(name="Person", source_table="persons", id_column="id", ontology_mapping="none", properties=[])
        ],
        relationships=[
            Relationship(
                name="KNOWS",
                source_table="knows",
                start_node="Person",
                start_key="p1",
                end_node="Person",
                end_key="p2",
            )
        ],
    )

    mock_client = mocker.Mock()
    mock_client.merge_relationships = mocker.AsyncMock()

    engine = ProjectionEngine(client=mock_client, resolver=resolver)
    job = GraphJob(id=uuid.uuid4(), manifest_path="test", status="RESOLVING")

    async with adapter:
        await engine.ingest_relationships(manifest, adapter, job)

    # 2 rows total (1 valid, 1 invalid) -> 1 created
    assert job.metrics["edges_created"] == 1.0


@pytest.mark.asyncio
async def test_entity_ingestion_failure(
    mocker: MockerFixture, manifest: ProjectionManifest, adapter: MockSourceAdapter, resolver: MockOntologyResolver
) -> None:
    mock_client = mocker.Mock()
    mock_client.merge_nodes = mocker.AsyncMock(side_effect=Exception("DB Fail"))

    engine = ProjectionEngine(client=mock_client, resolver=resolver)
    job = GraphJob(id=uuid.uuid4(), manifest_path="test", status="RESOLVING")

    async with adapter:
        with pytest.raises(Exception, match="DB Fail"):
            await engine.ingest_entities(manifest, adapter, job)


@pytest.mark.asyncio
async def test_relationship_ingestion_failure(
    mocker: MockerFixture, manifest: ProjectionManifest, adapter: MockSourceAdapter, resolver: MockOntologyResolver
) -> None:
    mock_client = mocker.Mock()
    mock_client.merge_relationships = mocker.AsyncMock(side_effect=Exception("DB Fail"))

    engine = ProjectionEngine(client=mock_client, resolver=resolver)
    job = GraphJob(id=uuid.uuid4(), manifest_path="test", status="RESOLVING")

    async with adapter:
        with pytest.raises(Exception, match="DB Fail"):
            await engine.ingest_relationships(manifest, adapter, job)
