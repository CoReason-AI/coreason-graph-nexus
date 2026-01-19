# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from collections.abc import Awaitable, Callable, Iterable, Iterator
from itertools import batched
from typing import Any, TypeVar

from coreason_graph_nexus.utils.logger import logger

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type (processed)


def process_and_batch(
    items: Iterable[T],
    processor: Callable[[T], R | None],
    consumer: Callable[[list[R]], Any],
    batch_size: int,
) -> int:
    """
    Processes an iterable of items, filters None values, batches them,
    and passes each batch to a consumer.

    Args:
        items: An iterable of input items.
        processor: A function that takes an item and returns a processed item or None to skip.
        consumer: A function that takes a list of processed items (a batch) and performs an action (e.g., DB write).
        batch_size: The size of each batch.

    Returns:
        The total number of processed items consumed.

    Raises:
        Exception: If the consumer raises an exception during batch processing.
    """
    processed_count = 0

    # Generator: Process items lazily
    processed_stream: Iterator[R] = (result for item in items if (result := processor(item)) is not None)

    # Batch and Consume
    for batch_tuple in batched(processed_stream, batch_size):
        batch_list = list(batch_tuple)

        try:
            consumer(batch_list)
            processed_count += len(batch_list)
        except Exception as e:
            logger.error(f"Failed to process batch of size {len(batch_list)}: {e}")
            raise

    return processed_count


async def process_and_batch_async(
    items: Iterable[T],
    processor: Callable[[T], R | None],
    consumer: Callable[[list[R]], Awaitable[Any]],
    batch_size: int,
) -> int:
    """
    Processes an iterable of items, filters None values, batches them,
    and passes each batch to an async consumer.

    Args:
        items: An iterable of input items.
        processor: A function that takes an item and returns a processed item or None to skip.
        consumer: An async function that takes a list of processed items (a batch).
        batch_size: The size of each batch.

    Returns:
        The total number of processed items consumed.
    """
    processed_count = 0

    # Generator: Process items lazily (Sync processor)
    processed_stream: Iterator[R] = (result for item in items if (result := processor(item)) is not None)

    # Batch and Consume
    for batch_tuple in batched(processed_stream, batch_size):
        batch_list = list(batch_tuple)

        try:
            await consumer(batch_list)
            processed_count += len(batch_list)
        except Exception as e:
            logger.error(f"Failed to process batch of size {len(batch_list)}: {e}")
            raise

    return processed_count
