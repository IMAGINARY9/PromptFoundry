"""Rate limiting utilities for LLM clients.

This module provides token bucket rate limiters for controlling
API request rates.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Token bucket rate limiter.

    Implements a token bucket algorithm for rate limiting.
    Thread-safe for async use.

    Attributes:
        capacity: Maximum tokens in bucket.
        refill_rate: Tokens added per second.
        tokens: Current available tokens.
        last_refill: Timestamp of last refill.
    """

    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(default_factory=time.monotonic)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        """Initialize tokens to capacity."""
        self.tokens = self.capacity

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    async def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds.
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            deficit = tokens - self.tokens
            wait_time = deficit / self.refill_rate

            return wait_time

    async def wait_and_acquire(self, tokens: float = 1.0) -> float:
        """Wait until tokens are available, then acquire.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Time waited in seconds.
        """
        wait_time = await self.acquire(tokens)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            async with self._lock:
                self._refill()
                self.tokens -= tokens
        return wait_time

    @property
    def available(self) -> float:
        """Return currently available tokens (without acquiring lock)."""
        return self.tokens


@dataclass
class RateLimiter:
    """Combined rate limiter for requests and tokens.

    Manages both request-per-minute and tokens-per-minute limits.

    Attributes:
        rpm: Requests per minute limit (0 = unlimited).
        tpm: Tokens per minute limit (0 = unlimited).
    """

    rpm: int = 0
    tpm: int = 0
    _request_bucket: TokenBucket | None = field(init=False, default=None)
    _token_bucket: TokenBucket | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initialize token buckets."""
        if self.rpm > 0:
            # Convert RPM to tokens/sec
            self._request_bucket = TokenBucket(
                capacity=float(self.rpm),
                refill_rate=self.rpm / 60.0,
            )
        if self.tpm > 0:
            self._token_bucket = TokenBucket(
                capacity=float(self.tpm),
                refill_rate=self.tpm / 60.0,
            )

    async def acquire_request(self, estimated_tokens: int = 0) -> float:
        """Acquire permission to make a request.

        Args:
            estimated_tokens: Estimated tokens for this request.

        Returns:
            Total time waited in seconds.
        """
        total_wait = 0.0

        if self._request_bucket:
            wait = await self._request_bucket.wait_and_acquire(1.0)
            total_wait += wait

        if self._token_bucket and estimated_tokens > 0:
            wait = await self._token_bucket.wait_and_acquire(float(estimated_tokens))
            total_wait += wait

        return total_wait

    @property
    def is_limited(self) -> bool:
        """Check if any rate limiting is enabled."""
        return self.rpm > 0 or self.tpm > 0
