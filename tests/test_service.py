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

from coreason_graph_nexus import Service, ServiceAsync
from coreason_graph_nexus.models import Entity, ProjectionManifest, PropertyMapping
from tests.conftest import MockOntologyResolver, MockSourceAdapter


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
        relationships=[],
    )


@pytest.fixture
def adapter() -> MockSourceAdapter:
    data: dict[str, list[dict[str, Any]]] = {"persons": [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]}
    return MockSourceAdapter(data)


@pytest.fixture
def resolver() -> MockOntologyResolver:
    return MockOntologyResolver({"1": "Person:1", "2": "Person:2"})


@pytest.mark.asyncio
async def test_service_async_lifecycle(mocker: MockerFixture) -> None:
    # Mock Neo4j driver
    mock_driver = mocker.AsyncMock()
    mocker.patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver)

    async with ServiceAsync() as svc:
        assert svc.neo4j is not None
        await svc.neo4j.verify_connectivity()

    mock_driver.close.assert_awaited()


@pytest.mark.asyncio
async def test_service_async_projection(
    mocker: MockerFixture, manifest: ProjectionManifest, adapter: MockSourceAdapter, resolver: MockOntologyResolver
) -> None:
    mock_driver = mocker.AsyncMock()
    mocker.patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver)

    # Mock execute_query to not fail
    mock_driver.execute_query.return_value = ([], None, None)

    async with ServiceAsync() as svc:
        job = await svc.run_projection(manifest, adapter, resolver, str(uuid.uuid4()))
        assert job.status == "COMPLETE"
        assert job.metrics["nodes_created"] == 2.0


def test_service_sync_facade(
    mocker: MockerFixture, manifest: ProjectionManifest, adapter: MockSourceAdapter, resolver: MockOntologyResolver
) -> None:
    mock_driver = mocker.AsyncMock()
    mocker.patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver)
    mock_driver.execute_query.return_value = ([], None, None)

    with Service() as svc:
        job = svc.run_projection(manifest, adapter, resolver, str(uuid.uuid4()))
        assert job.status == "COMPLETE"
        assert job.metrics["nodes_created"] == 2.0
