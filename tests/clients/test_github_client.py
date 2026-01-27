"""Tests for GitHub client core functionality using TDD approach."""

import pytest
from unittest.mock import AsyncMock, Mock, patch


class TestGitHubClientInitialization:
    """Test GitHubClient initialization."""

    @pytest.mark.parametrize("token,base_url,timeout,max_retries", [
        ("ghp_test_token", "https://api.github.com", 30, 3),
        ("ghp_another_token", "https://api.github.com", 60, 5),
        ("ghp_custom", "https://custom.github.com", 30, 3),
    ])
    def test_client_initialization_with_token(
        self, token, base_url, timeout, max_retries
    ):
        """Test GitHubClient can be initialized with a token."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(
            token=token,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        assert client.token == token
        assert client.base_url == base_url
        assert client.timeout == timeout
        assert client.max_retries == max_retries

    def test_client_initialization_with_custom_base_url(self):
        """Test GitHubClient can be initialized with a custom base URL."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(
            token="ghp_test",
            base_url="https://github.example.com",
        )

        assert client.base_url == "https://github.example.com"

    def test_client_initialization_defaults(self):
        """Test GitHubClient uses default values when not specified."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(token="ghp_test")

        assert client.token == "ghp_test"
        assert client.base_url == "https://api.github.com"
        assert client.timeout == 30
        assert client.max_retries == 3


@pytest.mark.asyncio
class TestGitHubClientMakeRequest:
    """Test GitHubClient._make_request method."""

    @pytest.mark.parametrize("method,endpoint", [
        ("GET", "/repos/owner/repo/issues"),
        ("POST", "/repos/owner/repo/issues"),
        ("PATCH", "/repos/owner/repo/issues/1"),
        ("DELETE", "/repos/owner/repo/issues/1"),
    ])
    async def test_make_request_adds_authentication_header(
        self, method, endpoint
    ):
        """Test _make_request adds authentication header."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(token="ghp_test_token")

        with patch.object(client, "_session") as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json.return_value = {}
            mock_response.text = "{}"
            mock_session.request = AsyncMock(return_value=mock_response)

            await client._make_request(method, endpoint)

            mock_session.request.assert_called_once()
            call_kwargs = mock_session.request.call_args.kwargs
            assert "Authorization" in call_kwargs.get("headers", {})


@pytest.mark.asyncio
class TestGitHubClientErrorHandling:
    """Test GitHubClient error handling in _make_request."""

    async def test_make_request_handles_200_ok_response(self):
        """Test _make_request handles 200 OK response correctly."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(token="ghp_test")

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json.return_value = {"id": 1, "title": "Test"}
        mock_response.text = '{"id": 1, "title": "Test"}'

        with patch.object(client, "_session") as mock_session:
            mock_session.request = AsyncMock(return_value=mock_response)

            result = await client._make_request("GET", "/test")

            assert result == {"id": 1, "title": "Test"}

    async def test_make_request_handles_404_not_found(self):
        """Test _make_request raises GitHubNotFoundError for 404."""
        from app.clients.github_client import GitHubClient, GitHubNotFoundError

        client = GitHubClient(token="ghp_test")

        mock_response = Mock()
        mock_response.status = 404
        mock_response.text = "Not Found"

        with patch.object(client, "_session") as mock_session:
            mock_session.request = AsyncMock(return_value=mock_response)

            with pytest.raises(GitHubNotFoundError) as exc_info:
                await client._make_request("GET", "/test")

            assert "Not Found" in str(exc_info.value) or "not found" in str(exc_info.value).lower()

    async def test_make_request_handles_401_unauthorized(self):
        """Test _make_request raises GitHubAuthenticationError for 401."""
        from app.clients.github_client import GitHubClient, GitHubAuthenticationError

        client = GitHubClient(token="ghp_test")

        mock_response = Mock()
        mock_response.status = 401
        mock_response.text = "Unauthorized"

        with patch.object(client, "_session") as mock_session:
            mock_session.request = AsyncMock(return_value=mock_response)

            with pytest.raises(GitHubAuthenticationError):
                await client._make_request("GET", "/test")

    async def test_make_request_handles_429_rate_limit(self):
        """Test _make_request raises GitHubRateLimitError for 429."""
        from app.clients.github_client import GitHubClient, GitHubRateLimitError

        client = GitHubClient(token="ghp_test")

        mock_response = Mock()
        mock_response.status = 429
        mock_response.text = "Rate limit exceeded"
        mock_response.headers = {"X-RateLimit-Reset": "1234567890"}

        with patch.object(client, "_session") as mock_session:
            mock_session.request = AsyncMock(return_value=mock_response)

            with pytest.raises(GitHubRateLimitError):
                await client._make_request("GET", "/test")

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    async def test_make_request_handles_5xx_errors(self, status_code):
        """Test _make_request raises GitHubAPIError for 5xx errors."""
        from app.clients.github_client import GitHubClient, GitHubAPIError

        client = GitHubClient(token="ghp_test")

        mock_response = Mock()
        mock_response.status = status_code
        mock_response.text = "Server Error"

        with patch.object(client, "_session") as mock_session:
            mock_session.request = AsyncMock(return_value=mock_response)

            with pytest.raises(GitHubAPIError):
                await client._make_request("GET", "/test")


@pytest.mark.asyncio
class TestGitHubClientRateLimiterIntegration:
    """Test GitHubClient integrates with rate limiter."""

    async def test_make_request_integrates_rate_limiter(self):
        """Test _make_request calls rate limiter.acquire()."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(token="ghp_test")

        with patch.object(client._rate_limiter, "acquire", new_callable=AsyncMock) as mock_acquire:
            with patch.object(client, "_session") as mock_session:
                mock_response = Mock()
                mock_response.status = 200
                mock_response.json.return_value = {}
                mock_response.text = "{}"
                mock_session.request = AsyncMock(return_value=mock_response)

                await client._make_request("GET", "/test")

                mock_acquire.assert_called_once()
