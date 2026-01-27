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

import pyarrow.parquet as pq

from coreason_graph_nexus.interfaces import SourceAdapter


class ParquetAdapter(SourceAdapter):
    """
    Adapter for reading Parquet files in a streaming fashion.
    """

    def connect(self) -> None:
        """
        No explicit connection is needed for Parquet files.
        """
        pass

    def disconnect(self) -> None:
        """
        No explicit disconnection is needed for Parquet files.
        """
        pass

    def read_table(self, table_name: str) -> Iterator[dict[str, Any]]:
        """
        Reads data from the specified Parquet file.

        Args:
            table_name: The file path of the Parquet file.

        Yields:
            A dictionary representing a single row (column_name -> value).
        """
        # Treat table_name as a file path
        parquet_file = pq.ParquetFile(table_name)

        # Iterate through the file using iter_batches to fetch small chunks
        # This ensures strictly streaming behavior and avoids loading the full file into RAM
        for batch in parquet_file.iter_batches(batch_size=10000):
            yield from batch.to_pylist()
