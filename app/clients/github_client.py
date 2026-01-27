"""GitHub API client for the GitHub Agent MCP integration.

This module provides exception classes and rate limiting for GitHub API interactions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class GitHubAPIError(Exception):
    """Base exception for all GitHub API errors.

    This exception should be raised when the GitHub API returns an error response.
    All specific GitHub exceptions inherit from this class.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the GitHub API error.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class GitHubRateLimitError(GitHubAPIError):
    """Exception raised when GitHub API rate limit is exceeded.

    This exception indicates that the client has made too many requests
    to the GitHub API and must wait before making more requests.
    """

    def __init__(
        self,
        message: str = "GitHub API rate limit exceeded",
        reset_time: int | None = None,
    ) -> None:
        """Initialize the rate limit error.

        Args:
            message: The error message.
            reset_time: Optional Unix timestamp when the rate limit will reset.
        """
        details: dict[str, Any] = {}
        if reset_time is not None:
            details["reset_time"] = reset_time
        super().__init__(message, details)
        self.reset_time = reset_time


class GitHubAuthenticationError(GitHubAPIError):
    """Exception raised for GitHub API authentication failures.

    This exception indicates that the provided authentication token
    is invalid, expired, or has insufficient permissions.
    """

    def __init__(self, message: str = "GitHub API authentication failed") -> None:
        """Initialize the authentication error.

        Args:
            message: The error message.
        """
        super().__init__(message)


class GitHubNotFoundError(GitHubAPIError):
    """Exception raised when a requested resource is not found.

    This exception indicates that the requested GitHub resource
    (issue, PR, repository, etc.) does not exist or is not accessible.
    """

    def __init__(self, message: str = "Resource not found") -> None:
        """Initialize the not found error.

        Args:
            message: The error message.
        """
        super().__init__(message)


class GitHubValidationError(GitHubAPIError):
    """Exception raised for GitHub API validation errors.

    This exception indicates that the request data was invalid
    or contained unexpected values.
    """

    def __init__(self, message: str = "Validation failed") -> None:
        """Initialize the validation error.

        Args:
            message: The error message.
        """
        super().__init__(message)


class GitHubRateLimiter:
    """Rate limiter for GitHub API requests using a sliding window.

    This class implements a sliding window rate limiting algorithm with a 1-hour window.
    It ensures that the number of requests does not exceed the specified limit
    within any 1-hour period.

    Attributes:
        requests_per_hour: Maximum number of requests allowed per hour.
        _requests: List of timestamps of recent requests.
        _lock: Async lock for thread-safe access.
    """

    def __init__(self, requests_per_hour: int = 5000) -> None:
        """Initialize the rate limiter.

        Args:
            requests_per_hour: Maximum number of requests allowed per hour.
                            Default is 5000 (GitHub's typical authenticated limit).
        """
        self.requests_per_hour = requests_per_hour
        self._requests: list[float] = []
        self._lock = asyncio.Lock()

    async def _cleanup_old_requests(self) -> None:
        """Remove requests that are older than 1 hour."""
        now = time.time()
        one_hour_ago = now - 3600
        self._requests = [t for t in self._requests if t > one_hour_ago]

    async def acquire(self) -> None:
        """Acquire permission to make a request.

        This method will wait if the rate limit has been exceeded.
        Once allowed, it records the request timestamp.

        Raises:
            GitHubRateLimitError: If the API indicates a stricter rate limit.
        """
        async with self._lock:
            await self._cleanup_old_requests()

            # Check if we're at the limit
            if len(self._requests) >= self.requests_per_hour:
                # Calculate how long to wait for the oldest request to expire
                oldest_request = self._requests[0]
                wait_time = oldest_request + 3600 - time.time()
                if wait_time > 0:
                    logger.warning(
                        f"Rate limit reached ({self.requests_per_hour} req/hour). "
                        f"Waiting {wait_time:.2f}s for oldest request to expire."
                    )
                    await asyncio.sleep(wait_time)
                    # Clean up after waiting
                    await self._cleanup_old_requests()

            # Record this request
            self._requests.append(time.time())

    def get_current_usage(self) -> int:
        """Get the current number of requests in the window.

        Returns:
            The number of requests made in the last hour.
        """
        now = time.time()
        one_hour_ago = now - 3600
        return len([t for t in self._requests if t > one_hour_ago])

    def get_remaining_requests(self) -> int:
        """Get the number of remaining requests in the window.

        Returns:
            The number of requests that can still be made this hour.
        """
        return max(0, self.requests_per_hour - self.get_current_usage())


class GitHubClient:
    """HTTP client for GitHub API with authentication and rate limiting.

    This client provides methods to interact with the GitHub API, handling
    authentication, rate limiting, and error responses.

    Attributes:
        token: GitHub personal access token for authentication.
        base_url: Base URL for GitHub API.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries for failed requests.
        _rate_limiter: Rate limiter instance.
        _session: HTTP session for making requests.
    """

    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """Initialize the GitHub client.

        Args:
            token: GitHub personal access token.
            base_url: Base URL for GitHub API.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._rate_limiter = GitHubRateLimiter()
        self._session: Any = None  # Will be aiohttp.ClientSession in production

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the GitHub API.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE).
            endpoint: API endpoint path.
            params: Query parameters.
            data: Request body data.

        Returns:
            The JSON response from the API.

        Raises:
            GitHubNotFoundError: For 404 responses.
            GitHubAuthenticationError: For 401 responses.
            GitHubRateLimitError: For 429 responses.
            GitHubAPIError: For other error responses.
            GitHubValidationError: For invalid request data.
        """
        await self._rate_limiter.acquire()

        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        }

        # In production, this would use aiohttp:
        # async with self._session.request(
        #     method, url, headers=headers, params=params, json=data
        # ) as response:
        #     return await self._handle_response(response)

        # For testing without aiohttp, we'll mock this behavior
        raise NotImplementedError(
            "GitHubClient._make_request requires aiohttp to be installed. "
            "Install with: pip install aiohttp"
        )

    async def _handle_response(self, response: Any) -> dict[str, Any]:
        """Handle HTTP response and raise appropriate exceptions.

        Args:
            response: HTTP response object.

        Returns:
            The JSON response data.

        Raises:
            GitHubNotFoundError: For 404 responses.
            GitHubAuthenticationError: For 401 responses.
            GitHubRateLimitError: For 429 responses.
            GitHubAPIError: For other error responses.
        """
        status = getattr(response, "status", response.get("status", 200))

        if status == 200:
            return await self._parse_json(response)
        elif status == 404:
            text = await getattr(response, "text", lambda: "Not Found")()
            raise GitHubNotFoundError(text)
        elif status == 401:
            text = await getattr(response, "text", lambda: "Unauthorized")()
            raise GitHubAuthenticationError(text)
        elif status == 429:
            headers = getattr(response, "headers", {})
            reset_time = headers.get("X-RateLimit-Reset")
            if reset_time:
                reset_time = int(reset_time)
            text = await getattr(response, "text", lambda: "Rate limit exceeded")()
            raise GitHubRateLimitError(text, reset_time)
        elif 400 <= status < 500:
            text = await getattr(response, "text", lambda: "Bad request")()
            raise GitHubValidationError(text)
        elif status >= 500:
            text = await getattr(response, "text", lambda: "Server error")()
            raise GitHubAPIError(f"GitHub API error {status}: {text}")

    async def _parse_json(self, response: Any) -> dict[str, Any]:
        """Parse JSON response with error handling.

        Args:
            response: HTTP response object.

        Returns:
            The parsed JSON data.

        Raises:
            GitHubAPIError: If JSON parsing fails.
        """
        try:
            return await response.json()
        except Exception as e:
            text = await getattr(response, "text", lambda: "")()
            raise GitHubAPIError(f"Failed to parse JSON response: {e}. Response: {text}")

    def _parse_issue(self, issue_data: dict[str, Any]) -> dict[str, Any]:
        """Parse GitHub API issue data into standardized format.

        Args:
            issue_data: Raw issue data from GitHub API.

        Returns:
            Standardized issue data dictionary.
        """
        return {
            "id": issue_data.get("id"),
            "number": issue_data.get("number"),
            "title": issue_data.get("title"),
            "body": issue_data.get("body"),
            "state": issue_data.get("state"),
            "author": issue_data.get("user", {}).get("login"),
            "assignees": [
                assignee.get("login") for assignee in issue_data.get("assignees", [])
            ],
            "labels": [
                label.get("name") for label in issue_data.get("labels", [])
            ],
            "created_at": issue_data.get("created_at"),
            "updated_at": issue_data.get("updated_at"),
            "url": issue_data.get("html_url"),
        }

    async def list_issues(
        self,
        owner: str,
        repo: str,
        state: str | None = None,
        assignee: str | None = None,
        labels: list[str] | None = None,
        milestone: str | None = None,
        page: int = 1,
        per_page: int = 30,
    ) -> list[dict[str, Any]]:
        """List issues in a repository.

        Args:
            owner: Repository owner username.
            repo: Repository name.
            state: Filter by issue state (open/closed).
            assignee: Filter by assignee username.
            labels: Filter by labels (comma-separated).
            milestone: Filter by milestone.
            page: Page number for pagination.
            per_page: Number of items per page.

        Returns:
            List of issue data dictionaries.

        Raises:
            GitHubAPIError: If the request fails.
        """
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if state is not None:
            params["state"] = state
        if assignee is not None:
            params["assignee"] = assignee
        if labels is not None:
            params["labels"] = ",".join(labels)
        if milestone is not None:
            params["milestone"] = milestone

        response = await self._make_request(
            "GET", f"/repos/{owner}/{repo}/issues", params=params
        )

        return [self._parse_issue(issue) for issue in response]

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new issue in a repository.

        Args:
            owner: Repository owner username.
            repo: Repository name.
            title: Issue title.
            body: Issue body content.
            labels: Issue labels.
            assignees: Usernames to assign to the issue.

        Returns:
            The created issue data dictionary.

        Raises:
            GitHubAPIError: If the request fails.
        """
        data: dict[str, Any] = {"title": title}
        if body is not None:
            data["body"] = body
        if labels is not None:
            data["labels"] = [{"name": label} for label in labels]
        if assignees is not None:
            data["assignees"] = assignees

        response = await self._make_request(
            "POST", f"/repos/{owner}/{repo}/issues", data=data
        )

        return self._parse_issue(response)

    async def update_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update an existing issue.

        Args:
            owner: Repository owner username.
            repo: Repository name.
            issue_number: The issue number.
            title: New issue title.
            body: New issue body.
            state: New issue state (open/closed).
            labels: New issue labels.
            assignees: New assignees.

        Returns:
            The updated issue data dictionary.

        Raises:
            GitHubAPIError: If the request fails.
        """
        data: dict[str, Any] = {}
        if title is not None:
            data["title"] = title
        if body is not None:
            data["body"] = body
        if state is not None:
            data["state"] = state
        if labels is not None:
            data["labels"] = [{"name": label} for label in labels]
        if assignees is not None:
            data["assignees"] = assignees

        response = await self._make_request(
            "PATCH", f"/repos/{owner}/{repo}/issues/{issue_number}", data=data
        )

        return self._parse_issue(response)

    async def close_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Close an issue.

        Args:
            owner: Repository owner username.
            repo: Repository name.
            issue_number: The issue number.
            comment: Optional comment to add when closing.

        Returns:
            The closed issue data dictionary.

        Raises:
            GitHubAPIError: If the request fails.
        """
        data: dict[str, Any] = {"state": "closed"}
        if comment is not None:
            data["body"] = comment

        response = await self._make_request(
            "PATCH", f"/repos/{owner}/{repo}/issues/{issue_number}", data=data
        )

        return self._parse_issue(response)

    async def close(self) -> None:
        """Close the HTTP session.

        This method should be called when the client is no longer needed.
        """
        if self._session is not None:
            await self._session.close()

    async def __aenter__(self) -> GitHubClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
