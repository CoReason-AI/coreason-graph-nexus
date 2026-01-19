# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

import contextvars
from collections.abc import AsyncIterator
from typing import Any

from coreason_graph_nexus.interfaces import OntologyResolver, SourceAdapter

# Context propagation example
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="unknown")


class MockSourceAdapter(SourceAdapter):
    """
    Mock Source Adapter for testing.
    """

    def __init__(self, data: dict[str, list[dict[str, Any]]]) -> None:
        self.data = data
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def read_table(self, table_name: str) -> AsyncIterator[dict[str, Any]]:
        if not self.connected:
            raise RuntimeError("Not connected")

        rows = self.data.get(table_name, [])
        for row in rows:
            yield row


class MockOntologyResolver(OntologyResolver):
    """
    Mock Ontology Resolver for testing.
    """

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    async def resolve(self, term: str) -> tuple[str | None, bool]:
        return self.mapping.get(term), True
