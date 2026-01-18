# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from unittest.mock import MagicMock

import pytest

from coreason_graph_nexus.adapters.neo4j_adapter import Neo4jClient


def test_batch_write_chunking(mock_driver: MagicMock) -> None:
    """Test that batch_write correctly chunks data using batched."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.return_value = ([], None, None)

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    # Create data larger than batch size
    data = [{"id": i} for i in range(25)]
    batch_size = 10

    client.batch_write("UNWIND $batch AS row ...", data, batch_size=batch_size)

    # Expect 3 calls: 10, 10, 5
    assert driver_instance.execute_query.call_count == 3

    # Verify payloads
    call_args_list = driver_instance.execute_query.call_args_list

    # Call 1
    args, kwargs = call_args_list[0]
    assert len(kwargs["parameters_"]["batch"]) == 10
    assert kwargs["parameters_"]["batch"][0]["id"] == 0

    # Call 2
    args, kwargs = call_args_list[1]
    assert len(kwargs["parameters_"]["batch"]) == 10
    assert kwargs["parameters_"]["batch"][0]["id"] == 10

    # Call 3
    args, kwargs = call_args_list[2]
    assert len(kwargs["parameters_"]["batch"]) == 5
    assert kwargs["parameters_"]["batch"][0]["id"] == 20


def test_batch_write_iterable(mock_driver: MagicMock) -> None:
    """Test that batch_write accepts a generator/iterable."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.return_value = ([], None, None)

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    data_gen = ({"id": i} for i in range(5))

    client.batch_write("UNWIND...", data_gen, batch_size=2)

    assert driver_instance.execute_query.call_count == 3  # 2, 2, 1


def test_merge_nodes(mock_driver: MagicMock) -> None:
    """Test merge_nodes calls batch_write with correct query."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.return_value = ([], None, None)

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    data = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
    client.merge_nodes("Person", data, merge_keys=["id"], batch_size=10)

    driver_instance.execute_query.assert_called_once()
    args, kwargs = driver_instance.execute_query.call_args
    query = args[0]

    assert "UNWIND $batch AS row MERGE (n:`Person` { `id`: row.`id` }) SET n += row" in query
    assert kwargs["parameters_"]["batch"] == data


def test_merge_relationships(mock_driver: MagicMock) -> None:
    """Test merge_relationships calls batch_write with correct query."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.return_value = ([], None, None)

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    data = [{"start": 1, "end": 2, "prop": "val"}]
    client.merge_relationships(
        "A", "start", "B", "end", "REL", data, start_node_prop="sid", end_node_prop="eid"
    )

    driver_instance.execute_query.assert_called_once()
    args, kwargs = driver_instance.execute_query.call_args
    query = args[0]

    assert "MATCH (source:`A` { `sid`: row.`start` })" in query
    assert "MATCH (target:`B` { `eid`: row.`end` })" in query
    assert "MERGE (source)-[r:`REL`]->(target)" in query
    assert kwargs["parameters_"]["batch"] == data


def test_batch_write_invalid_size(mock_driver: MagicMock) -> None:
    """Test validation of batch_size."""
    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
    with pytest.raises(ValueError, match="Batch size must be positive"):
        client.batch_write("Q", [], batch_size=0)


def test_batch_write_empty(mock_driver: MagicMock) -> None:
    """Test batch_write with empty data logs info and doesn't execute."""
    driver_instance = mock_driver.return_value
    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    client.batch_write("Q", [])
    driver_instance.execute_query.assert_not_called()


def test_batch_write_exception(mock_driver: MagicMock) -> None:
    """Test exception handling during batch execution."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.side_effect = Exception("DB Fail")

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    with pytest.raises(Exception, match="DB Fail"):
        client.batch_write("Q", [{"a": 1}])


def test_merge_nodes_invalid_keys(mock_driver: MagicMock) -> None:
    """Test merge_nodes validation for empty keys."""
    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))
    with pytest.raises(ValueError, match="merge_keys must not be empty"):
        client.merge_nodes("L", [], merge_keys=[])


def test_batch_write_generator_exception(mock_driver: MagicMock) -> None:
    """Test that if the input generator raises an exception, it propagates."""
    driver_instance = mock_driver.return_value
    driver_instance.execute_query.return_value = ([], None, None)

    client = Neo4jClient("bolt://localhost:7687", ("user", "pass"))

    def fail_gen():
        yield {"id": 1}
        raise ValueError("Generator Failed")

    # We use batch_size=1 so the first yield is processed before the error
    with pytest.raises(ValueError, match="Generator Failed"):
        client.batch_write("Q", fail_gen(), batch_size=1)

    # Should have called execute_query once for the first item
    assert driver_instance.execute_query.call_count == 1
