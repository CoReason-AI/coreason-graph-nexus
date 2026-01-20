# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

import functools
from typing import Any, Callable, Protocol

from coreason_graph_nexus.interfaces import OntologyResolver
from coreason_graph_nexus.utils.logger import logger


class RedisClientProtocol(Protocol):
    """Protocol for Redis client to allow type checking without hard dependency."""

    def get(self, key: str) -> bytes | None: ...
    def setex(self, key: str, time: int, value: str) -> Any: ...


def cached_resolver(
    ttl: int = 86400,
) -> Callable[[Callable[[Any, str], str | None]], Callable[[Any, str], tuple[str | None, bool]]]:
    """
    Decorator to cache the result of an ontology resolution method in Redis.

    Expected class structure:
    class MyClass:
        self.redis_client: Redis

    Args:
        ttl: Time to live in seconds (default: 24h).

    Returns:
        A decorator function that wraps the resolution method.
    """

    def decorator(
        func: Callable[[Any, str], str | None],
    ) -> Callable[[Any, str], tuple[str | None, bool]]:
        @functools.wraps(func)
        def wrapper(self: Any, term: str) -> tuple[str | None, bool]:
            if not hasattr(self, "redis_client") or self.redis_client is None:
                logger.warning("Redis client not available in self.redis_client. Skipping cache.")
                return func(self, term), False

            key = f"resolve:{term}"
            try:
                cached = self.redis_client.get(key)
                if cached:
                    logger.debug(f"Cache hit for '{term}': {cached}")
                    # Redis returns bytes, decode to str
                    val = cached.decode("utf-8") if isinstance(cached, bytes) else str(cached)
                    return val, True
            except Exception as e:
                logger.error(f"Redis get failed for key {key}: {e}")

            # Call the actual function
            result = func(self, term)

            if result:
                try:
                    self.redis_client.setex(key, ttl, result)
                    logger.debug(f"Cache set for '{term}' -> '{result}' (TTL: {ttl}s)")
                except Exception as e:
                    logger.error(f"Redis setex failed for key {key}: {e}")

            return result, False

        return wrapper

    return decorator


class CodexClient:
    """
    Client for the external Coreason Codex service.

    Since external services are not available in this environment,
    this class serves as a placeholder/interface for the integration.
    """

    def lookup_concept(self, term: str) -> str | None:
        """
        Looks up a concept in the Codex.

        Args:
            term: The term to look up.

        Returns:
            The canonical ID if found, else None.
        """
        # In a real implementation, this would make an HTTP request.
        # For now, we simulate a miss or a dummy value.
        logger.warning(f"Codex lookup simulated for '{term}' (Not Implemented)")
        return None


class RedisOntologyResolver(OntologyResolver):
    """
    Ontology Resolver that uses Redis for caching and Codex as the authority.
    """

    def __init__(self, redis_client: Any, codex_client: CodexClient | None = None) -> None:
        """
        Initialize the resolver.

        Args:
            redis_client: A redis.Redis instance (or mock).
            codex_client: A CodexClient instance. If None, creates a default one.
        """
        self.redis_client = redis_client
        self.codex_client = codex_client or CodexClient()

    @cached_resolver(ttl=86400)  # type: ignore[arg-type]
    def resolve(self, term: str) -> tuple[str | None, bool]:
        """
        Resolves a term to a canonical ID.

        Checks Redis first (via decorator). If miss, calls Codex.

        Args:
            term: The source string to resolve.

        Returns:
            A tuple of (canonical identifier (str) | None, is_cache_hit (bool)).
        """
        return self.codex_client.lookup_concept(term)  # type: ignore[return-value]
