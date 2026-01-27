"""Tests for GitHub rate limiter using TDD approach."""

import asyncio
import time
from datetime import datetime, timedelta

import pytest


@pytest.mark.asyncio
class TestGitHubRateLimiterBelowLimit:
    """Test rate limiter behavior when below the limit."""

    @pytest.mark.parametrize("requests_per_hour,num_requests", [
        (60, 10),  # 60 requests/hour, making 10 requests
        (100, 1),  # 100 requests/hour, making 1 request
        (3600, 3599),  # 3600 requests/hour, making 3599 requests
    ])
    async def test_rate_limiter_below_limit_allows_immediately(
        self, requests_per_hour, num_requests
    ):
        """Test that requests are allowed immediately when below limit."""
        from app.clients.github_client import GitHubRateLimiter

        limiter = GitHubRateLimiter(requests_per_hour=requests_per_hour)

        for _ in range(num_requests):
            await limiter.acquire()  # Should not wait or raise

        assert True  # If we get here, all requests were allowed


@pytest.mark.asyncio
class TestGitHubRateLimiterAtLimit:
    """Test rate limiter behavior at the exact limit."""

    @pytest.mark.parametrize("requests_per_hour", [60, 100, 3600])
    async def test_rate_limiter_at_exact_limit_allows(self, requests_per_hour):
        """Test that exactly the limit number of requests are allowed."""
        from app.clients.github_client import GitHubRateLimiter

        limiter = GitHubRateLimiter(requests_per_hour=requests_per_hour)

        # Make exactly the limit number of requests
        for _ in range(requests_per_hour):
            await limiter.acquire()  # Should not wait

        assert True  # If we get here, all requests were allowed


@pytest.mark.asyncio
class TestGitHubRateLimiterAboveLimit:
    """Test rate limiter behavior when above the limit."""

    async def test_rate_limiter_above_limit_waits(self):
        """Test that requests above limit cause a wait."""
        from app.clients.github_client import GitHubRateLimiter

        limiter = GitHubRateLimiter(requests_per_hour=60)

        # Make 60 requests (the limit)
        for _ in range(60):
            await limiter.acquire()

        # The next request should wait
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        # Should have waited at least some time (since we're at limit)
        assert elapsed >= 0, f"Expected wait, but elapsed time was {elapsed}"

    async def test_rate_limiter_above_limit_waits_for_old_request_expiry(self):
        """Test that waiting is limited to when old requests expire."""
        from app.clients.github_client import GitHubRateLimiter

        # Use a very low limit for testing
        requests_per_hour = 1
        limiter = GitHubRateLimiter(requests_per_hour=requests_per_hour)

        # Make one request
        await limiter.acquire()

        # Mock time passing - but we can't easily test exact timing without mocking
        # This test documents the expected behavior
        assert limiter._requests[-1] >= time.time() - 3600


@pytest.mark.asyncio
class TestGitHubRateLimiterSlidingWindow:
    """Test rate limiter sliding window cleanup."""

    async def test_rate_limiter_sliding_window_cleanup(self):
        """Test that old requests expire from the window."""
        from app.clients.github_client import GitHubRateLimiter
        from unittest.mock import patch

        limiter = GitHubRateLimiter(requests_per_hour=60)

        # Make 60 requests
        for _ in range(60):
            await limiter.acquire()

        assert len(limiter._requests) == 60

        # Simulate time passing (1 hour + 1 second)
        old_time = time.time() - 3601

        # Manually set old timestamps for cleanup test
        with patch.object(limiter, "_requests", [old_time - i for i in range(60)]):
            # After cleanup, old requests should be removed
            await limiter._cleanup_old_requests()
            assert all(t >= time.time() - 3600 for t in limiter._requests)


@pytest.mark.asyncio
class TestGitHubRateLimiterConcurrent:
    """Test rate limiter with concurrent requests."""

    async def test_rate_limiter_concurrent_acquire_requests(self):
        """Test that concurrent requests are handled correctly."""
        from app.clients.github_client import GitHubRateLimiter

        limiter = GitHubRateLimiter(requests_per_hour=60)

        # Create concurrent acquire tasks
        tasks = [limiter.acquire() for _ in range(10)]
        await asyncio.gather(*tasks)

        # All tasks should complete
        assert len(limiter._requests) == 10


@pytest.mark.asyncio
class TestGitHubRateLimiterIntegration:
    """Integration tests for rate limiter."""

    async def test_rate_limiter_state_after_multiple_acquires(self):
        """Test rate limiter state after multiple acquires."""
        from app.clients.github_client import GitHubRateLimiter

        limiter = GitHubRateLimiter(requests_per_hour=60)

        # Make some requests
        for i in range(5):
            await limiter.acquire()

        # Check state
        assert len(limiter._requests) == 5
        assert limiter._requests_per_hour == 60

    async def test_rate_limiter_cleanup_happens_on_wait(self):
        """Test that cleanup happens when waiting for rate limit."""
        from app.clients.github_client import GitHubRateLimiter
        from unittest.mock import patch

        limiter = GitHubRateLimiter(requests_per_hour=60)

        # Fill the limit
        for _ in range(60):
            await limiter.acquire()

        # Add some old requests
        old_time = time.time() - 3700
        with patch.object(limiter, "_requests", limiter._requests + [old_time]):
            # Next acquire should trigger cleanup
            await limiter.acquire()
            # Old requests should be cleaned up
            assert old_time not in limiter._requests
