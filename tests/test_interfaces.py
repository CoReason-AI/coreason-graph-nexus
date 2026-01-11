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

import pytest

from coreason_graph_nexus.interfaces import SourceAdapter


class MockSourceAdapter(SourceAdapter):
    """A concrete implementation of SourceAdapter for testing purposes."""

    def __init__(self, data: dict[str, list[dict[str, Any]]]) -> None:
        self.data = data
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_table(self, table_name: str) -> Iterator[dict[str, Any]]:
        if not self.connected:
            raise ConnectionError("Not connected")
        yield from self.data.get(table_name, [])


def test_source_adapter_context_manager() -> None:
    data = {"users": [{"id": 1, "name": "Alice"}]}
    adapter = MockSourceAdapter(data)

    assert not adapter.connected
    with adapter as client:
        assert client.connected
        assert client is adapter
    assert not adapter.connected


def test_source_adapter_read_table() -> None:
    data = {
        "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "empty": [],
    }
    with MockSourceAdapter(data) as client:
        # Test generic read
        users = list(client.read_table("users"))
        assert len(users) == 2
        assert users[0]["name"] == "Alice"

        # Test empty table
        empty = list(client.read_table("empty"))
        assert len(empty) == 0

        # Test non-existent table
        missing = list(client.read_table("missing"))
        assert len(missing) == 0


def test_source_adapter_not_connected_error() -> None:
    adapter = MockSourceAdapter({})
    with pytest.raises(ConnectionError, match="Not connected"):
        list(adapter.read_table("users"))
