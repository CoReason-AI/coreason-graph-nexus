# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from unittest.mock import MagicMock

import pytest

from coreason_graph_nexus.ontology import CodexClient, RedisOntologyResolver, cached_resolver


class TestRedisOntologyResolver:
    """
    Tests for the Redis-backed Ontology Resolver.
    """

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        """Mock the Redis client."""
        return MagicMock()

    @pytest.fixture
    def mock_codex_client(self) -> MagicMock:
        """Mock the Codex client."""
        return MagicMock(spec=CodexClient)

    def test_cache_hit(self, mock_redis: MagicMock, mock_codex_client: MagicMock) -> None:
        """
        Test that a cache hit returns the value from Redis without calling Codex.
        """
        # Setup: Redis has the key
        mock_redis.get.return_value = b"RxNorm:123"  # Redis returns bytes

        resolver = RedisOntologyResolver(redis_client=mock_redis, codex_client=mock_codex_client)
        result = resolver.resolve("Tylenol")

        assert result == "RxNorm:123"
        mock_redis.get.assert_called_once_with("resolve:Tylenol")
        mock_codex_client.lookup_concept.assert_not_called()

    def test_cache_miss_found_in_codex(self, mock_redis: MagicMock, mock_codex_client: MagicMock) -> None:
        """
        Test that a cache miss calls Codex, returns the result, and caches it.
        """
        # Setup: Redis miss
        mock_redis.get.return_value = None
        # Setup: Codex found
        mock_codex_client.lookup_concept.return_value = "RxNorm:456"

        resolver = RedisOntologyResolver(redis_client=mock_redis, codex_client=mock_codex_client)
        result = resolver.resolve("Advil")

        assert result == "RxNorm:456"
        mock_redis.get.assert_called_once_with("resolve:Advil")
        mock_codex_client.lookup_concept.assert_called_once_with("Advil")
        # Verify it writes back to Redis with 24h TTL (86400 seconds)
        mock_redis.setex.assert_called_once_with("resolve:Advil", 86400, "RxNorm:456")

    def test_cache_miss_not_found_in_codex(self, mock_redis: MagicMock, mock_codex_client: MagicMock) -> None:
        """
        Test that if Codex returns None, we return None and do NOT cache None.
        """
        # Setup: Redis miss
        mock_redis.get.return_value = None
        # Setup: Codex miss
        mock_codex_client.lookup_concept.return_value = None

        resolver = RedisOntologyResolver(redis_client=mock_redis, codex_client=mock_codex_client)
        result = resolver.resolve("UnknownDrug")

        assert result is None
        mock_codex_client.lookup_concept.assert_called_once_with("UnknownDrug")
        mock_redis.setex.assert_not_called()

    def test_decorator_usage(self, mock_redis: MagicMock) -> None:
        """
        Test the standalone decorator if we choose to expose it directly.
        """

        # Define a class that uses the decorator
        class MyService:
            def __init__(self, redis: MagicMock) -> None:
                self.redis_client = redis

            @cached_resolver(ttl=100)
            def resolve(self, term: str) -> str:
                return "Computed:" + term

        service = MyService(mock_redis)

        # Scenario 1: Cache Miss
        mock_redis.get.return_value = None
        result = service.resolve("Test")
        assert result == "Computed:Test"
        mock_redis.setex.assert_called_with("resolve:Test", 100, "Computed:Test")

        # Scenario 2: Cache Hit
        mock_redis.get.return_value = b"Cached:Test"
        result = service.resolve("Test")
        assert result == "Cached:Test"

    def test_missing_redis_client(self) -> None:
        """
        Test that the decorator proceeds without caching if redis_client is missing or None.
        """

        class MyServiceNoRedis:
            def __init__(self) -> None:
                self.redis_client = None

            @cached_resolver()
            def resolve(self, term: str) -> str:
                return "Value:" + term

        service = MyServiceNoRedis()
        result = service.resolve("Test")
        assert result == "Value:Test"

    def test_redis_get_exception(self, mock_redis: MagicMock, mock_codex_client: MagicMock) -> None:
        """
        Test that if Redis.get fails, it logs error and calls the function.
        """
        mock_redis.get.side_effect = Exception("Connection Refused")
        mock_codex_client.lookup_concept.return_value = "Result"

        resolver = RedisOntologyResolver(redis_client=mock_redis, codex_client=mock_codex_client)
        result = resolver.resolve("Term")

        assert result == "Result"
        mock_codex_client.lookup_concept.assert_called_once_with("Term")
        # Should still try to set if get failed? Yes, unless logic prevents.
        # Implementation: try get -> catch -> call func -> try set.
        mock_redis.setex.assert_called_once()

    def test_redis_setex_exception(self, mock_redis: MagicMock, mock_codex_client: MagicMock) -> None:
        """
        Test that if Redis.setex fails, it logs error and returns result.
        """
        mock_redis.get.return_value = None
        mock_redis.setex.side_effect = Exception("Write Failed")
        mock_codex_client.lookup_concept.return_value = "Result"

        resolver = RedisOntologyResolver(redis_client=mock_redis, codex_client=mock_codex_client)
        result = resolver.resolve("Term")

        assert result == "Result"

    def test_codex_client_lookup(self) -> None:
        """
        Test the CodexClient placeholder directly to cover its log line.
        """
        client = CodexClient()
        result = client.lookup_concept("Something")
        assert result is None

    def test_resolve_special_characters(self, mock_redis: MagicMock, mock_codex_client: MagicMock) -> None:
        """
        Test that terms with special characters are handled correctly in Redis keys.
        """
        mock_redis.get.return_value = None
        mock_codex_client.lookup_concept.return_value = "ID:123"

        resolver = RedisOntologyResolver(redis_client=mock_redis, codex_client=mock_codex_client)
        term = "Drug A + B (Old)"

        result = resolver.resolve(term)

        assert result == "ID:123"
        # The key should preserve the term exactly
        mock_redis.get.assert_called_once_with("resolve:Drug A + B (Old)")
        mock_redis.setex.assert_called_once_with("resolve:Drug A + B (Old)", 86400, "ID:123")

    def test_empty_string_input(self, mock_redis: MagicMock, mock_codex_client: MagicMock) -> None:
        """
        Test behavior with empty string input.
        """
        mock_redis.get.return_value = None
        mock_codex_client.lookup_concept.return_value = None

        resolver = RedisOntologyResolver(redis_client=mock_redis, codex_client=mock_codex_client)
        result = resolver.resolve("")

        assert result is None
        mock_redis.get.assert_called_once_with("resolve:")

    def test_codex_exception_propagation(self, mock_redis: MagicMock, mock_codex_client: MagicMock) -> None:
        """
        Test that exceptions raised by the backend (Codex) are propagated.
        """
        mock_redis.get.return_value = None
        mock_codex_client.lookup_concept.side_effect = ValueError("Backend Error")

        resolver = RedisOntologyResolver(redis_client=mock_redis, codex_client=mock_codex_client)

        with pytest.raises(ValueError, match="Backend Error"):
            resolver.resolve("Term")

        # Should not have tried to cache anything
        mock_redis.setex.assert_not_called()

    def test_redis_returns_non_bytes(self, mock_redis: MagicMock, mock_codex_client: MagicMock) -> None:
        """
        Test handling when Redis returns a non-bytes object (e.g., str or int).
        """
        # Scenario 1: Redis returns a string (already decoded)
        mock_redis.get.return_value = "RxNorm:789"

        resolver = RedisOntologyResolver(redis_client=mock_redis, codex_client=mock_codex_client)
        result = resolver.resolve("DrugX")

        assert result == "RxNorm:789"

        # Scenario 2: Redis returns an integer (unlikely but robust code should handle it)
        mock_redis.get.return_value = 12345
        result = resolver.resolve("DrugY")
        assert result == "12345"
