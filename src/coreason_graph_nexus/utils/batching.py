# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from collections.abc import AsyncIterable, Callable
from typing import Any, TypeVar

from coreason_graph_nexus.utils.logger import logger

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type (processed)


async def process_and_batch(
    items: AsyncIterable[T],
    processor: Callable[[T], R | None],
    consumer: Callable[[list[R]], Any],
    batch_size: int,
) -> int:
    """
    Processes an async iterable of items, filters None values, batches them,
    and passes each batch to an async consumer.

    Args:
        items: An async iterable of input items.
        processor: A function that takes an item and returns a processed item or None to skip.
        consumer: An async function that takes a list of processed items (a batch) and performs an
                  action (e.g., DB write).
        batch_size: The size of each batch.

    Returns:
        The total number of processed items consumed.

    Raises:
        Exception: If the consumer raises an exception during batch processing.
    """
    processed_count = 0
    current_batch: list[R] = []

    async for item in items:
        result = processor(item)
        if result is None:
            continue

        current_batch.append(result)

        if len(current_batch) >= batch_size:
            try:
                await consumer(current_batch)
                processed_count += len(current_batch)
                current_batch = []
            except Exception as e:
                logger.error(f"Failed to process batch of size {len(current_batch)}: {e}")
                raise

    # Process remaining items
    if current_batch:
        try:
            await consumer(current_batch)
            processed_count += len(current_batch)
        except Exception as e:
            logger.error(f"Failed to process final batch of size {len(current_batch)}: {e}")
            raise

    return processed_count
