# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from collections.abc import AsyncIterator
from typing import Any

import pytest

from coreason_graph_nexus.utils.batching import process_and_batch


async def async_generator(items: list[Any]) -> AsyncIterator[Any]:
    for i in items:
        yield i


@pytest.mark.asyncio
async def test_process_and_batch_success() -> None:
    items = [1, 2, 3, 4, 5]
    processed_batches = []

    def processor(item: int) -> int:
        return item * 2

    async def consumer(batch: list[int]) -> None:
        processed_batches.append(batch)

    count = await process_and_batch(async_generator(items), processor, consumer, batch_size=2)

    assert count == 5
    assert processed_batches == [[2, 4], [6, 8], [10]]


@pytest.mark.asyncio
async def test_process_and_batch_async_processor() -> None:
    items = [1]

    async def processor(item: int) -> int:
        return item

    async def consumer(batch: list[int]) -> None:
        pass

    count = await process_and_batch(async_generator(items), processor, consumer, batch_size=1)
    assert count == 1


@pytest.mark.asyncio
async def test_process_and_batch_consumer_error() -> None:
    items = [1, 2]

    def processor(item: int) -> int:
        return item

    async def consumer(batch: list[int]) -> None:
        raise ValueError("Boom")

    with pytest.raises(ValueError, match="Boom"):
        await process_and_batch(async_generator(items), processor, consumer, batch_size=1)


@pytest.mark.asyncio
async def test_process_and_batch_consumer_error_final_batch() -> None:
    items = [1]

    def processor(item: int) -> int:
        return item

    async def consumer(batch: list[int]) -> None:
        raise ValueError("Boom Final")

    with pytest.raises(ValueError, match="Boom Final"):
        await process_and_batch(async_generator(items), processor, consumer, batch_size=10)
