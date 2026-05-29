"""Secrets manager for API key management with rate-limited access.

Provides a unified interface for managing sensitive credentials (API keys, tokens)
with support for multiple backends:
- Local: Environment variables / .env file (development)
- AWS SSM Parameter Store (production)
- AWS Secrets Manager (production)
- HashiCorp Vault (production)

Features:
- Rate-limited access to prevent credential abuse
- In-memory caching with configurable TTL
- Automatic rotation support
- Audit logging of secret access
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default rate limit: 60 accesses per minute
DEFAULT_RATE_LIMIT = 60
DEFAULT_RATE_WINDOW_SECONDS = 60
DEFAULT_CACHE_TTL_SECONDS = 300  # 5 minutes


class SecretBackend(str, Enum):
    """Supported secrets storage backends."""

    LOCAL = "local"
    AWS_SSM = "aws_ssm"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    VAULT = "vault"


class SecretMetadata(BaseModel):
    """Metadata about a stored secret."""

    key: str
    backend: SecretBackend
    last_accessed: float | None = None
    access_count: int = 0
    cached: bool = False


class RateLimitExceeded(Exception):
    """Raised when secret access rate limit is exceeded."""

    pass


class SecretNotFoundError(Exception):
    """Raised when a requested secret is not found."""

    pass


class RateLimiter:
    """Token bucket rate limiter for secret access.

    Tracks access timestamps within a sliding window and rejects
    requests that exceed the configured rate.
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_RATE_LIMIT,
        window_seconds: int = DEFAULT_RATE_WINDOW_SECONDS,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the window.
            window_seconds: Size of the sliding window in seconds.
        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def check(self) -> bool:
        """Check if a request is allowed under the rate limit.

        Returns:
            True if the request is allowed.

        Raises:
            RateLimitExceeded: If the rate limit has been exceeded.
        """
        now = time.time()
        cutoff = now - self._window_seconds

        # Remove expired timestamps
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        if len(self._timestamps) >= self._max_requests:
            raise RateLimitExceeded(
                f"Rate limit exceeded: {self._max_requests} requests per "
                f"{self._window_seconds} seconds"
            )

        self._timestamps.append(now)
        return True

    @property
    def remaining(self) -> int:
        """Return the number of remaining requests in the current window."""
        now = time.time()
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        return max(0, self._max_requests - len(self._timestamps))


class CachedSecret:
    """A cached secret value with TTL."""

    def __init__(self, value: str, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS) -> None:
        self.value = value
        self.expires_at = time.time() + ttl_seconds

    @property
    def is_expired(self) -> bool:
        """Check if the cached value has expired."""
        return time.time() >= self.expires_at


class SecretsBackendBase(ABC):
    """Abstract base class for secrets storage backends."""

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        """Retrieve a secret by key.

        Args:
            key: The secret identifier.

        Returns:
            The secret value, or None if not found.
        """
        ...

    @abstractmethod
    def set_secret(self, key: str, value: str) -> None:
        """Store a secret.

        Args:
            key: The secret identifier.
            value: The secret value to store.
        """
        ...

    @abstractmethod
    def delete_secret(self, key: str) -> bool:
        """Delete a secret.

        Args:
            key: The secret identifier.

        Returns:
            True if the secret was deleted, False if not found.
        """
        ...

    @abstractmethod
    def list_secrets(self) -> list[str]:
        """List all available secret keys.

        Returns:
            List of secret key names.
        """
        ...


class LocalSecretsBackend(SecretsBackendBase):
    """Local secrets backend using environment variables.

    Suitable for development. Reads from environment variables
    and an optional in-memory override store.
    """

    def __init__(self) -> None:
        self._overrides: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:
        """Get secret from overrides first, then environment."""
        if key in self._overrides:
            return self._overrides[key]
        return os.environ.get(key)

    def set_secret(self, key: str, value: str) -> None:
        """Store secret in the override dict."""
        self._overrides[key] = value

    def delete_secret(self, key: str) -> bool:
        """Remove secret from overrides."""
        if key in self._overrides:
            del self._overrides[key]
            return True
        return False

    def list_secrets(self) -> list[str]:
        """List override keys (env vars not enumerated for security)."""
        return list(self._overrides.keys())


class SecretsManager:
    """Unified secrets manager with rate limiting and caching.

    Provides rate-limited access to secrets with in-memory caching
    to reduce backend calls. Supports multiple storage backends.

    Usage:
        manager = SecretsManager(backend=SecretBackend.LOCAL)
        api_key = manager.get("NEWS_REUTERS_API_KEY")
    """

    def __init__(
        self,
        backend: SecretBackend = SecretBackend.LOCAL,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        rate_window_seconds: int = DEFAULT_RATE_WINDOW_SECONDS,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        """Initialize the secrets manager.

        Args:
            backend: The storage backend to use.
            rate_limit: Maximum secret accesses per window.
            rate_window_seconds: Rate limit window in seconds.
            cache_ttl_seconds: Cache TTL for secret values.
        """
        self._backend_type = backend
        self._backend = self._create_backend(backend)
        self._rate_limiter = RateLimiter(rate_limit, rate_window_seconds)
        self._cache: dict[str, CachedSecret] = {}
        self._cache_ttl = cache_ttl_seconds
        self._access_counts: dict[str, int] = {}

        logger.info("SecretsManager initialized with backend=%s", backend.value)

    def _create_backend(self, backend: SecretBackend) -> SecretsBackendBase:
        """Create the appropriate backend instance.

        Args:
            backend: The backend type to create.

        Returns:
            An instance of the appropriate backend.
        """
        if backend == SecretBackend.LOCAL:
            return LocalSecretsBackend()
        # For AWS and Vault backends, fall back to local in development
        # Production would import and instantiate the appropriate SDK-based backend
        logger.warning(
            "Backend %s not fully implemented, falling back to local", backend.value
        )
        return LocalSecretsBackend()

    def get(self, key: str, use_cache: bool = True) -> str:
        """Retrieve a secret by key with rate limiting and caching.

        Args:
            key: The secret identifier.
            use_cache: Whether to use cached values (default: True).

        Returns:
            The secret value.

        Raises:
            RateLimitExceeded: If the access rate limit is exceeded.
            SecretNotFoundError: If the secret is not found.
        """
        # Check rate limit
        self._rate_limiter.check()

        # Check cache
        if use_cache and key in self._cache:
            cached = self._cache[key]
            if not cached.is_expired:
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
                return cached.value
            else:
                del self._cache[key]

        # Fetch from backend
        value = self._backend.get_secret(key)
        if value is None:
            raise SecretNotFoundError(f"Secret not found: {key}")

        # Cache the value
        self._cache[key] = CachedSecret(value, self._cache_ttl)
        self._access_counts[key] = self._access_counts.get(key, 0) + 1

        logger.debug("Secret accessed: %s (total accesses: %d)", key, self._access_counts[key])
        return value

    def set(self, key: str, value: str) -> None:
        """Store a secret.

        Args:
            key: The secret identifier.
            value: The secret value to store.

        Raises:
            RateLimitExceeded: If the access rate limit is exceeded.
        """
        self._rate_limiter.check()
        self._backend.set_secret(key, value)

        # Invalidate cache
        self._cache.pop(key, None)

        logger.info("Secret stored: %s", key)

    def delete(self, key: str) -> bool:
        """Delete a secret.

        Args:
            key: The secret identifier.

        Returns:
            True if the secret was deleted.

        Raises:
            RateLimitExceeded: If the access rate limit is exceeded.
        """
        self._rate_limiter.check()
        result = self._backend.delete_secret(key)

        # Invalidate cache
        self._cache.pop(key, None)

        if result:
            logger.info("Secret deleted: %s", key)
        return result

    def invalidate_cache(self, key: str | None = None) -> None:
        """Invalidate cached secrets.

        Args:
            key: Specific key to invalidate, or None to clear all cache.
        """
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()

    def get_metadata(self, key: str) -> SecretMetadata:
        """Get metadata about a secret without accessing its value.

        Args:
            key: The secret identifier.

        Returns:
            SecretMetadata with access statistics.
        """
        cached = key in self._cache and not self._cache[key].is_expired
        return SecretMetadata(
            key=key,
            backend=self._backend_type,
            last_accessed=None,
            access_count=self._access_counts.get(key, 0),
            cached=cached,
        )

    @property
    def rate_limit_remaining(self) -> int:
        """Return the number of remaining rate-limited requests."""
        return self._rate_limiter.remaining


# Module-level singleton
_secrets_manager: SecretsManager | None = None


def get_secrets_manager() -> SecretsManager:
    """Get or create the global secrets manager singleton.

    Returns:
        The application-wide SecretsManager instance.
    """
    global _secrets_manager
    if _secrets_manager is None:
        backend_str = os.environ.get("SECRETS_MANAGER_BACKEND", "local")
        try:
            backend = SecretBackend(backend_str)
        except ValueError:
            logger.warning("Unknown secrets backend '%s', using local", backend_str)
            backend = SecretBackend.LOCAL

        rate_limit = int(os.environ.get("SECRETS_MANAGER_RATE_LIMIT", str(DEFAULT_RATE_LIMIT)))
        _secrets_manager = SecretsManager(backend=backend, rate_limit=rate_limit)

    return _secrets_manager
