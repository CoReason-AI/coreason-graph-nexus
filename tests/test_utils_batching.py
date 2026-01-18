# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

import pytest
from coreason_graph_nexus.utils.batching import process_and_batch

def test_process_and_batch_basic():
    """Test basic processing and batching logic."""
    items = range(10)
    processed_batches = []

    def processor(x):
        return x * 2

    def consumer(batch):
        processed_batches.append(batch)

    count = process_and_batch(items, processor, consumer, batch_size=3)

    assert count == 10
    assert len(processed_batches) == 4
    assert processed_batches[0] == [0, 2, 4]
    assert processed_batches[1] == [6, 8, 10]
    assert processed_batches[2] == [12, 14, 16]
    assert processed_batches[3] == [18]

def test_process_and_batch_filtering():
    """Test that None values returned by processor are skipped."""
    items = range(10)
    processed_batches = []

    def processor(x):
        return x if x % 2 == 0 else None  # Keep evens

    def consumer(batch):
        processed_batches.append(batch)

    count = process_and_batch(items, processor, consumer, batch_size=2)

    # Evens: 0, 2, 4, 6, 8 (Total 5)
    assert count == 5
    assert len(processed_batches) == 3
    assert processed_batches[0] == [0, 2]
    assert processed_batches[1] == [4, 6]
    assert processed_batches[2] == [8]

def test_process_and_batch_exception_propagation():
    """Test that exceptions in consumer are propagated."""
    items = [1, 2, 3]

    def processor(x):
        return x

    def consumer(batch):
        raise ValueError("Consumer failed")

    with pytest.raises(ValueError, match="Consumer failed"):
        process_and_batch(items, processor, consumer, batch_size=2)

def test_process_and_batch_empty_input():
    """Test behavior with empty input."""
    items = []
    processed_batches = []

    process_and_batch(items, lambda x: x, lambda b: processed_batches.append(b), batch_size=5)

    assert len(processed_batches) == 0

def test_process_and_batch_all_filtered():
    """Test when all items are filtered out."""
    items = [1, 3, 5]
    processed_batches = []

    def processor(x):
        return None

    count = process_and_batch(items, processor, lambda b: processed_batches.append(b), batch_size=2)

    assert count == 0
    assert len(processed_batches) == 0

def test_process_and_batch_logging_on_error(mocker):
    """Test that logger.error is called when consumer fails."""
    # We mock the logger to verify it is called
    mock_logger = mocker.patch("coreason_graph_nexus.utils.batching.logger")

    items = [1]

    def consumer(batch):
        raise ValueError("Boom")

    with pytest.raises(ValueError, match="Boom"):
        process_and_batch(items, lambda x: x, consumer, batch_size=1)

    mock_logger.error.assert_called_once()
    assert "Failed to process batch" in mock_logger.error.call_args[0][0]
