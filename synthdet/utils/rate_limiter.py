"""Token bucket rate limiter for API calls."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Thread-safe token bucket rate limiter.

    Tokens refill at ``requests_per_minute / 60`` per second.
    Burst capacity is ``requests_per_minute // 10`` (minimum 1).
    A rate of 0 disables limiting entirely.
    """

    def __init__(self, requests_per_minute: float = 600.0) -> None:
        self.requests_per_minute = requests_per_minute
        if requests_per_minute <= 0:
            self._disabled = True
            self._max_tokens = 0.0
            self._tokens = 0.0
            self._refill_rate = 0.0
        else:
            self._disabled = False
            self._max_tokens = max(1.0, requests_per_minute / 10)
            self._tokens = self._max_tokens
            self._refill_rate = requests_per_minute / 60.0
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 60.0) -> bool:
        """Block until a token is available. Returns False on timeout."""
        if self._disabled:
            return True

        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now
