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
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest_mock import MockerFixture

from coreason_graph_nexus.adapters.parquet_adapter import ParquetAdapter


@pytest.fixture
def parquet_file(tmp_path: Path) -> str:
    file_path = tmp_path / "test.parquet"
    data = [pa.array([1, 2, 3, 4, 5]), pa.array(["a", "b", "c", "d", "e"])]
    batch = pa.RecordBatch.from_arrays(data, names=["id", "value"])
    table = pa.Table.from_batches([batch])
    pq.write_table(table, file_path)
    return str(file_path)


@pytest.fixture
def large_parquet_file(tmp_path: Path) -> str:
    file_path = tmp_path / "large_test.parquet"
    # Create 20,000 rows to ensure we test iterating over batches (default batch size is 10,000)
    ids = range(20000)
    values = [str(i) for i in ids]
    data = [pa.array(ids), pa.array(values)]
    batch = pa.RecordBatch.from_arrays(data, names=["id", "value"])
    table = pa.Table.from_batches([batch])
    pq.write_table(table, file_path)
    return str(file_path)


def test_parquet_adapter_read(parquet_file: str) -> None:
    """Test reading a simple parquet file."""
    adapter = ParquetAdapter()
    rows = list(adapter.read_table(parquet_file))
    assert len(rows) == 5
    assert rows[0] == {"id": 1, "value": "a"}
    assert rows[4] == {"id": 5, "value": "e"}


def test_parquet_adapter_streaming(large_parquet_file: str) -> None:
    """Test reading a large file to ensure all rows are returned."""
    adapter = ParquetAdapter()
    rows = list(adapter.read_table(large_parquet_file))
    assert len(rows) == 20000
    assert rows[0]["id"] == 0
    assert rows[19999]["id"] == 19999


def test_parquet_adapter_streaming_calls(mocker: MockerFixture) -> None:
    """Test that iter_batches is called with the correct batch size."""
    adapter = ParquetAdapter()

    # Mock pq.ParquetFile
    mock_pf_cls = mocker.patch("coreason_graph_nexus.adapters.parquet_adapter.pq.ParquetFile")
    mock_pf_instance = mock_pf_cls.return_value

    # Mock iter_batches to return an empty iterator so the loop finishes immediately
    mock_pf_instance.iter_batches.return_value = iter([])

    list(adapter.read_table("dummy_path"))

    mock_pf_cls.assert_called_with("dummy_path")
    mock_pf_instance.iter_batches.assert_called_with(batch_size=10000)


def test_context_manager(parquet_file: str) -> None:
    """Test that the adapter works as a context manager."""
    with ParquetAdapter() as adapter:
        rows = list(adapter.read_table(parquet_file))
        assert len(rows) == 5


def test_file_not_found() -> None:
    """Test behavior when the file does not exist."""
    adapter = ParquetAdapter()
    # pyarrow raises FileNotFound or ArrowIOError depending on version/OS
    # We use pytest.raises(Exception, match=...) to satisfy B017 and cover both cases
    with pytest.raises((FileNotFoundError, pa.ArrowIOError)):
        list(adapter.read_table("non_existent_file.parquet"))


def test_empty_file(tmp_path: Path) -> None:
    """Test reading an empty parquet file."""
    file_path = tmp_path / "empty.parquet"
    schema = pa.schema([("id", pa.int64())])
    # Write a file with schema but no rows
    with pq.ParquetWriter(file_path, schema):
        pass

    adapter = ParquetAdapter()
    rows = list(adapter.read_table(str(file_path)))
    assert len(rows) == 0


def test_invalid_file(tmp_path: Path) -> None:
    """Test reading a file that is not a valid parquet file."""
    file_path = tmp_path / "invalid.parquet"
    file_path.write_text("not a parquet file")

    adapter = ParquetAdapter()
    # pyarrow raises ArrowInvalid for invalid files
    with pytest.raises(pa.ArrowInvalid):
        list(adapter.read_table(str(file_path)))


def test_parquet_diverse_types(tmp_path: Path) -> None:
    """Test reading a parquet file with various data types including nulls."""
    file_path = tmp_path / "types.parquet"
    data: list[Any] = [
        pa.array([1, 2, None, 4]),
        pa.array(["a", None, "c", "d"]),
        pa.array([True, False, True, None]),
        pa.array([[1, 2], [3], None, []]),
    ]
    batch = pa.RecordBatch.from_arrays(data, names=["int_col", "str_col", "bool_col", "list_col"])
    table = pa.Table.from_batches([batch])
    pq.write_table(table, file_path)

    adapter = ParquetAdapter()
    rows = list(adapter.read_table(str(file_path)))

    assert len(rows) == 4
    # Check row with nulls
    assert rows[2] == {
        "int_col": None,
        "str_col": "c",
        "bool_col": True,
        "list_col": None,
    }
    # Check list type
    assert rows[0]["list_col"] == [1, 2]
    # Check empty list
    assert rows[3]["list_col"] == []


def test_parquet_batch_boundary(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test streaming behavior with batch sizes at boundaries."""
    file_path = tmp_path / "boundary.parquet"
    # Create 15 rows
    data = [pa.array(range(15))]
    batch = pa.RecordBatch.from_arrays(data, names=["id"])
    table = pa.Table.from_batches([batch])
    pq.write_table(table, file_path)

    adapter = ParquetAdapter()

    # We can't easily control the batch_size passed to iter_batches from the outside
    # without mocking. So we spy on the call to ensure logic holds.

    # However, since we are testing the adapter's logic which hardcodes 10000,
    # we can't test "different batch sizes" without modifying the code or mocking ParquetFile.
    # So let's mock ParquetFile to verify we can handle whatever iter_batches yields.

    mock_pf_cls = mocker.patch("coreason_graph_nexus.adapters.parquet_adapter.pq.ParquetFile")
    mock_pf = mock_pf_cls.return_value

    # Simulate iter_batches returning batches of size 1
    # We yield 15 batches of size 1
    mock_batches = [pa.RecordBatch.from_arrays([pa.array([i])], names=["id"]) for i in range(15)]
    mock_pf.iter_batches.return_value = iter(mock_batches)

    rows = list(adapter.read_table(str(file_path)))
    assert len(rows) == 15
    assert rows[0]["id"] == 0
    assert rows[14]["id"] == 14


def test_complex_workflow_aggregation(tmp_path: Path) -> None:
    """
    Simulate a workflow where we aggregate data from a file.
    This mimics a 'Projection Engine' calculating stats.
    """
    file_path = tmp_path / "agg.parquet"
    values = range(100)
    data = [pa.array(values)]
    batch = pa.RecordBatch.from_arrays(data, names=["val"])
    table = pa.Table.from_batches([batch])
    pq.write_table(table, file_path)

    adapter = ParquetAdapter()
    total_sum = 0
    count = 0

    for row in adapter.read_table(str(file_path)):
        total_sum += row["val"]
        count += 1

    assert count == 100
    assert total_sum == sum(range(100))


def test_complex_workflow_multiple_files(tmp_path: Path) -> None:
    """
    Simulate processing multiple files sequentially.
    """
    files = []
    expected_ids: list[int] = []

    # Create 3 files
    for i in range(3):
        p = tmp_path / f"part_{i}.parquet"
        ids = range(i * 10, (i + 1) * 10)
        expected_ids.extend(ids)
        data = [pa.array(ids)]
        batch = pa.RecordBatch.from_arrays(data, names=["id"])
        table = pa.Table.from_batches([batch])
        pq.write_table(table, p)
        files.append(str(p))

    adapter = ParquetAdapter()
    processed_ids = []

    # Sequential processing
    for f in files:
        for row in adapter.read_table(f):
            processed_ids.append(row["id"])

    assert len(processed_ids) == 30
    assert processed_ids == expected_ids
