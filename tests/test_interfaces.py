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

    def __init__(
        self,
        data: dict[str, list[dict[str, Any]]],
        simulate_connect_error: bool = False,
        simulate_disconnect_error: bool = False,
        simulate_stream_error_at: int | None = None,
    ) -> None:
        self.data = data
        self.connected = False
        self.disconnect_called = False
        self.simulate_connect_error = simulate_connect_error
        self.simulate_disconnect_error = simulate_disconnect_error
        self.simulate_stream_error_at = simulate_stream_error_at

    def connect(self) -> None:
        if self.simulate_connect_error:
            raise ConnectionError("Failed to connect")
        self.connected = True
        self.disconnect_called = False

    def disconnect(self) -> None:
        if self.simulate_disconnect_error:
            raise ConnectionError("Failed to disconnect")
        self.connected = False
        self.disconnect_called = True

    def read_table(self, table_name: str) -> Iterator[dict[str, Any]]:
        if not self.connected:
            raise ConnectionError("Not connected")

        rows = self.data.get(table_name, [])
        for i, row in enumerate(rows):
            if self.simulate_stream_error_at is not None and i == self.simulate_stream_error_at:
                raise RuntimeError("Stream interrupted")
            yield row


def test_source_adapter_context_manager() -> None:
    data = {"users": [{"id": 1, "name": "Alice"}]}
    adapter = MockSourceAdapter(data)

    assert not adapter.connected
    with adapter as client:
        assert client.connected
        assert client is adapter
        # Ensure we didn't call disconnect yet
        assert not client.disconnect_called
    # After exit, connected should be false, and disconnect should have been called
    assert not adapter.connected
    assert adapter.disconnect_called


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


def test_context_manager_exception_safety() -> None:
    """
    Edge Case: Ensure disconnect is called even if an exception occurs
    inside the 'with' block.
    """
    adapter = MockSourceAdapter({})
    with pytest.raises(ValueError, match="Something went wrong"):
        with adapter:
            assert adapter.connected
            raise ValueError("Something went wrong")

    # Disconnect must have been called during cleanup
    assert not adapter.connected
    assert adapter.disconnect_called


def test_lazy_evaluation() -> None:
    """
    Edge Case: Verify that read_table is lazy (returns a generator)
    and doesn't yield data until iterated.
    """
    data = {"large_table": [{"id": i} for i in range(100)]}
    with MockSourceAdapter(data) as client:
        iterator = client.read_table("large_table")
        # Check it is an iterator, not a list
        assert isinstance(iterator, Iterator)
        # Pull one item
        first = next(iterator)
        assert first["id"] == 0
        # The rest haven't been consumed yet (conceptually, though difficult to prove strictly in mock)


def test_streaming_error_propagation() -> None:
    """
    Complex Scenario: Simulate a failure occurring mid-stream (e.g., network cut)
    while reading a table.
    """
    data = {"users": [{"id": 1}, {"id": 2}, {"id": 3}]}
    # Fail at index 1 (the second item)
    adapter = MockSourceAdapter(data, simulate_stream_error_at=1)

    with adapter as client:
        iterator = client.read_table("users")
        assert next(iterator) == {"id": 1}

        with pytest.raises(RuntimeError, match="Stream interrupted"):
            next(iterator)

    # Cleanup should still happen
    assert adapter.disconnect_called


def test_reusability() -> None:
    """
    Edge Case: Verify the adapter can be reused (Connect -> Disconnect -> Connect).
    """
    adapter = MockSourceAdapter({})

    # First use
    with adapter:
        assert adapter.connected
    assert not adapter.connected

    # Second use
    with adapter:
        assert adapter.connected
    assert not adapter.connected
