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
