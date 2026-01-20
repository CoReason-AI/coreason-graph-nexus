# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType
from typing import Any, Self


class SourceAdapter(ABC):
    """
    Abstract Interface for Data Source Adapters.

    This interface defines the contract for reading tabular data from upstream
    sources (e.g., SQL Databases, Parquet files, API endpoints) to be projected
    into the Graph.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establishes the connection to the data source."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Closes the connection to the data source."""
        ...

    @abstractmethod
    def read_table(self, table_name: str) -> Iterator[dict[str, Any]]:
        """
        Reads data from the specified table or view.

        Args:
            table_name: The name of the table/view to read from.

        Yields:
            A dictionary representing a single row (column_name -> value).
        """
        ...

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.disconnect()


class OntologyResolver(ABC):
    """
    Abstract Interface for Ontology Resolution.

    The resolver is responsible for mapping source terms (e.g., "Tylenol")
    to canonical concepts (e.g., "RxNorm:123") in the Knowledge Graph.
    """

    @abstractmethod
    def resolve(self, term: str) -> tuple[str | None, bool]:
        """
        Resolves a source term to a canonical identifier.

        Args:
            term: The source string to resolve.

        Returns:
            A tuple of (canonical identifier (str) | None, is_cache_hit (bool)).
        """
        ...
