"""Tests for synthdet.utils.rate_limiter."""

from __future__ import annotations

import time

from synthdet.utils.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_acquire_within_limit(self):
        """Immediate acquisition when tokens are available."""
        rl = RateLimiter(requests_per_minute=600.0)
        assert rl.acquire(timeout=1.0) is True

    def test_acquire_blocks_when_exhausted(self):
        """After exhausting burst tokens, acquire still succeeds after refill."""
        rl = RateLimiter(requests_per_minute=600.0)
        # Drain burst capacity
        burst = int(600 / 10)
        for _ in range(burst):
            assert rl.acquire(timeout=0.5) is True
        # Next acquire should block briefly but succeed (refill rate is 10/s)
        start = time.monotonic()
        assert rl.acquire(timeout=2.0) is True
        elapsed = time.monotonic() - start
        assert elapsed < 2.0

    def test_tokens_refill_over_time(self):
        """Tokens replenish at the correct rate."""
        rl = RateLimiter(requests_per_minute=60.0)  # 1 per second
        # Drain
        rl._tokens = 0.0
        rl._last_refill = time.monotonic()
        time.sleep(0.15)
        # Should have ~0.15 tokens, not enough
        # But acquire with 2s timeout should succeed
        assert rl.acquire(timeout=2.0) is True

    def test_burst_allowed(self):
        """Burst up to max_burst without blocking."""
        rl = RateLimiter(requests_per_minute=600.0)
        # max_burst = 600/10 = 60
        count = 0
        start = time.monotonic()
        for _ in range(60):
            if rl.acquire(timeout=0.01):
                count += 1
        elapsed = time.monotonic() - start
        assert count == 60
        assert elapsed < 1.0  # Should be near-instant

    def test_timeout_returns_false(self):
        """Returns False when timeout exceeded."""
        rl = RateLimiter(requests_per_minute=60.0)  # 1/s, burst=6
        # Drain all tokens
        for _ in range(6):
            rl.acquire(timeout=0.01)
        # Very short timeout — should fail
        assert rl.acquire(timeout=0.05) is False

    def test_unlimited_rate(self):
        """requests_per_minute=0 disables limiting."""
        rl = RateLimiter(requests_per_minute=0)
        for _ in range(100):
            assert rl.acquire(timeout=0.01) is True
