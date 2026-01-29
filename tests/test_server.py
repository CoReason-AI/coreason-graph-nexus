# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from coreason_graph_nexus.server import app


@pytest.fixture
def mock_redis():
    with patch("redis.Redis.from_url") as mock:
        client_mock = MagicMock()
        client_mock.ping.return_value = True
        mock.return_value = client_mock
        yield client_mock


@pytest.fixture
def mock_service():
    with patch("coreason_graph_nexus.server.ServiceAsync") as mock:
        service_instance = AsyncMock()
        service_instance.__aenter__.return_value = service_instance
        service_instance.__aexit__.return_value = None

        # Mock internal client for health check
        service_instance._client = AsyncMock()
        service_instance._client.execute_query.return_value = [{"count(n)": 10}]

        mock.return_value = service_instance
        yield service_instance


@pytest.fixture
def client(mock_redis, mock_service):
    # TestClient triggers lifespan
    with TestClient(app) as c:
        yield c


def test_health_check(client, mock_redis, mock_service):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["redis"] == "ok"
    assert data["neo4j"] == "ok"

    mock_redis.ping.assert_called()
    mock_service._client.execute_query.assert_called_with("MATCH (n) RETURN count(n) LIMIT 1")


def test_health_degraded(client, mock_redis, mock_service):
    # Test Redis failure
    mock_redis.ping.side_effect = Exception("Redis Down")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["redis"] == "error: Redis Down"
    assert response.json()["status"] == "degraded"

    # Restore Redis, break Neo4j
    mock_redis.ping.side_effect = None
    mock_service._client.execute_query.side_effect = Exception("Neo4j Down")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["neo4j"] == "error: Neo4j Down"
    assert response.json()["status"] == "degraded"


def test_project_ingest(client, mock_service):
    manifest_data = {
        "version": "1.0",
        "source_connection": "dummy_conn",
        "entities": [
            {
                "name": "Person",
                "source_table": "people.parquet",
                "id_column": "id",
                "ontology_mapping": "None",
                "properties": [{"source": "name", "target": "name"}]
            }
        ],
        "relationships": [
            {
                "name": "KNOWS",
                "source_table": "knows.parquet",
                "start_node": "Person",
                "start_key": "pid1",
                "end_node": "Person",
                "end_key": "pid2"
            }
        ]
    }

    payload = {
        "manifest": manifest_data,
        "source_base_path": "/tmp/data"
    }

    response = client.post("/project/ingest", json=payload)
    assert response.status_code == 200
    job = response.json()
    assert job["status"] == "PROJECTING"

    # Background task runs
    mock_service.ingest_entities.assert_called_once()
    mock_service.ingest_relationships.assert_called_once()

    # Check if manifest was updated with base path
    call_args_ent = mock_service.ingest_entities.call_args
    manifest_arg_ent = call_args_ent[0][0]
    assert "/tmp/data/people.parquet" in manifest_arg_ent.entities[0].source_table

    call_args_rel = mock_service.ingest_relationships.call_args
    manifest_arg_rel = call_args_rel[0][0]
    assert "/tmp/data/knows.parquet" in manifest_arg_rel.relationships[0].source_table


def test_project_ingest_failure(client, mock_service):
    # In background tasks, if exception occurs, TestClient might raise it if not handled?
    # But I catch Exception in the background task and log it.
    # So TestClient should NOT raise.

    manifest_data = {
        "version": "1.0",
        "source_connection": "dummy",
        "entities": [],
        "relationships": []
    }
    mock_service.ingest_entities.side_effect = Exception("Ingest Failed")

    # We patch logger to verify error is logged
    with patch("coreason_graph_nexus.server.logger") as mock_logger:
        response = client.post("/project/ingest", json={"manifest": manifest_data})
        assert response.status_code == 200
        assert response.json()["status"] == "PROJECTING"

        # Verification that background task ran and failed
        mock_service.ingest_entities.assert_called()
        mock_logger.error.assert_called()
        # Ensure correct error message
        args, _ = mock_logger.error.call_args
        assert "Ingest Failed" in str(args[0])


def test_compute_analysis(client, mock_service):
    request_data = {
        "center_node_id": "123",
        "algorithm": "pagerank",
        "depth": 2,
        "write_property": "rank"
    }

    mock_service.run_analysis.return_value = {"123": 0.5}

    response = client.post("/compute/analysis", json=request_data)
    assert response.status_code == 200
    assert response.json() == {"123": 0.5}

    mock_service.run_analysis.assert_called_once()


def test_compute_analysis_errors(client, mock_service):
    request_data = {
        "center_node_id": "123",
        "algorithm": "pagerank"
    }

    # ValueError
    mock_service.run_analysis.side_effect = ValueError("Bad Input")
    response = client.post("/compute/analysis", json=request_data)
    assert response.status_code == 400
    assert "Bad Input" in response.json()["detail"]

    # NotImplementedError
    mock_service.run_analysis.side_effect = NotImplementedError("Algo Missing")
    response = client.post("/compute/analysis", json=request_data)
    assert response.status_code == 501

    # Generic Exception
    mock_service.run_analysis.side_effect = Exception("Crash")
    response = client.post("/compute/analysis", json=request_data)
    assert response.status_code == 500


def test_predict_links(client, mock_service):
    request_data = {
        "method": "semantic",
        "source_label": "Person",
        "target_label": "Movie",
        "threshold": 0.8
    }

    response = client.post("/predict/links", json=request_data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    mock_service.predict_links.assert_called_once()


def test_predict_links_errors(client, mock_service):
    request_data = {
        "method": "semantic",
        "source_label": "Person",
        "target_label": "Movie"
    }

    mock_service.predict_links.side_effect = ValueError("Invalid Config")
    response = client.post("/predict/links", json=request_data)
    assert response.status_code == 400

    mock_service.predict_links.side_effect = Exception("Crash")
    response = client.post("/predict/links", json=request_data)
    assert response.status_code == 500


def test_lifespan_redis_failure():
    with patch("redis.Redis.from_url") as mock_redis:
        mock_client = MagicMock()
        mock_client.ping.side_effect = Exception("Connection Failed")
        mock_redis.return_value = mock_client

        with pytest.raises(RuntimeError, match="Redis connection failed"):
             with TestClient(app):
                 pass
