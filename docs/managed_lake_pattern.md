# Managed Lake Architectural Pattern

The Graph Nexus adopts the "Managed Lake" architectural pattern to efficiently ingest massive, governed datasets (e.g., Microsoft Graph Data Connect snapshots) without the overhead of loading entire datasets into memory. This approach leverages modern tooling and the Parquet file format to achieve high performance and scalability.

## Overview

The core principle of the "Managed Lake" pattern is to treat the data lake (specifically, Parquet files) as a first-class citizen in the data processing pipeline. Instead of extracting data into an intermediate store or loading it fully into application memory, the Graph Nexus reads directly from the source files in a streaming fashion.

### Key Benefits

1.  **Zero-Copy Streaming:** Data is read in small batches directly from the Parquet files. This minimizes memory usage and avoids the need for large, expensive compute instances to handle large datasets.
2.  **Scalability:** By processing data in streams, the system can handle datasets that are significantly larger than the available RAM.
3.  **Governance:** Direct ingestion from governed storage locations ensures that data lineage and security controls are maintained.

## Implementation: ParquetAdapter

The `ParquetAdapter` is the key component enabling this pattern. It implements the `SourceAdapter` interface and utilizes the `pyarrow` library to interact with Parquet files.

### How it Works

1.  **Direct File Access:** The adapter treats the `table_name` as a direct file path to the Parquet file.
2.  **Batch Processing:** It uses `pyarrow.parquet.ParquetFile.iter_batches()` to read data in configurable chunks (defaulting to 10,000 rows).
3.  **Streaming Interface:** The adapter yields individual rows as dictionaries, allowing the Projection Engine to process them one by one or in small groups, maintaining a low memory footprint.

### Usage

To use the `ParquetAdapter`, instantiate it and pass the path to your Parquet file to the `read_table` method:

```python
from coreason_graph_nexus.adapters import ParquetAdapter

adapter = ParquetAdapter()
for row in adapter.read_table("path/to/your/data.parquet"):
    process(row)
```

## Dependency Modernization

This implementation relies on `pyarrow` (version 23.0.0 or later), which provides robust support for the Parquet format and efficient memory management. This alignment with modern standards ensures that the Graph Nexus remains compatible with the evolving data ecosystem.
