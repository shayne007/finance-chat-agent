"""Tests for GitHub client issue operations using TDD approach."""

import pytest
from unittest.mock import AsyncMock, Mock, patch


@pytest.mark.asyncio
class TestGitHubClientListIssues:
    """Test GitHubClient.list_issues method."""

    @pytest.mark.parametrize("filters,expected_params", [
        ({"state": "open"}, {"state": "open"}),
        ({"assignee": "testuser"}, {"assignee": "testuser"}),
        ({"labels": ["bug", "enhancement"]}, {"labels": "bug,enhancement"}),
        ({"milestone": "v1.0"}, {"milestone": "v1.0"}),
        ({"state": "closed", "assignee": "user"}, {"state": "closed", "assignee": "user"}),
    ])
    async def test_list_issues_with_various_filters(
        self, filters, expected_params
    ):
        """Test list_issues with various filters."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(token="ghp_test")

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[
            {
                "id": 1,
                "number": 100,
                "title": "Test Issue",
                "state": "open",
                "user": {"login": "testuser"},
                "assignees": [],
                "labels": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "html_url": "https://github.com/test/repo/issues/100",
            }
        ])

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response.json.return_value

            result = await client.list_issues("owner", "repo", **filters)

            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args.kwargs
            assert call_kwargs["method"] == "GET"
            assert call_kwargs["endpoint"] == "/repos/owner/repo/issues"
            for key, value in expected_params.items():
                assert call_kwargs["params"].get(key) == value

    async def test_list_issues_with_pagination(self):
        """Test list_issues with pagination parameters."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(token="ghp_test")

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[])

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response.json.return_value

            await client.list_issues("owner", "repo", page=2, per_page=50)

            call_kwargs = mock_request.call_args.kwargs
            assert call_kwargs["params"]["page"] == 2
            assert call_kwargs["params"]["per_page"] == 50

    async def test_list_issues_returns_empty_list_when_no_issues(self):
        """Test list_issues returns empty list when no issues exist."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(token="ghp_test")

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = []

            result = await client.list_issues("owner", "repo")

            assert result == []

    async def test_list_issues_parses_response_into_github_issue_models(self):
        """Test list_issues parses response into GitHubIssue models."""
        from app.clients.github_client import GitHubClient
        from app.models.github import GitHubIssue

        client = GitHubClient(token="ghp_test")

        mock_response = [
            {
                "id": 1,
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "user": {"login": "testuser"},
                "assignees": [{"login": "assignee1"}],
                "labels": [{"name": "bug"}],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "html_url": "https://github.com/test/repo/issues/100",
            }
        ]

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.list_issues("owner", "repo")

            assert len(result) == 1
            assert isinstance(result[0], GitHubIssue)
            assert result[0].id == 1
            assert result[0].number == 100
            assert result[0].title == "Test Issue"
            assert result[0].author == "testuser"
            assert result[0].assignees == ["assignee1"]
            assert result[0].labels == ["bug"]


@pytest.mark.asyncio
class TestGitHubClientCreateIssue:
    """Test GitHubClient.create_issue method."""

    @pytest.mark.parametrize("data,expected_body", [
        (
            {"title": "New Issue", "body": "Description"},
            {"title": "New Issue", "body": "Description"}
        ),
        (
            {"title": "Another Issue"},
            {"title": "Another Issue"}
        ),
        (
            {
                "title": "Issue with labels",
                "body": "Description",
                "labels": ["bug"],
                "assignees": ["user1", "user2"],
            },
            {
                "title": "Issue with labels",
                "body": "Description",
                "labels": ["bug"],
                "assignees": ["user1", "user2"],
            },
        ),
    ])
    async def test_create_issue_with_various_data(self, data, expected_body):
        """Test create_issue with various inputs."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(token="ghp_test")

        mock_response = {
            "id": 2,
            "number": 101,
            "title": data["title"],
            "body": data.get("body"),
            "state": "open",
            "user": {"login": "testuser"},
            "assignees": data.get("assignees", []),
            "labels": [{"name": l} for l in data.get("labels", [])],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/test/repo/issues/101",
        }

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.create_issue("owner", "repo", **data)

            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args.kwargs
            assert call_kwargs["method"] == "POST"
            assert call_kwargs["endpoint"] == "/repos/owner/repo/issues"
            assert call_kwargs["data"] == expected_body

    async def test_create_issue_returns_github_issue_model(self):
        """Test create_issue returns GitHubIssue model."""
        from app.clients.github_client import GitHubClient
        from app.models.github import GitHubIssue

        client = GitHubClient(token="ghp_test")

        mock_response = {
            "id": 2,
            "number": 101,
            "title": "Created Issue",
            "state": "open",
            "user": {"login": "testuser"},
            "assignees": [],
            "labels": [],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/test/repo/issues/101",
        }

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.create_issue("owner", "repo", title="Test")

            assert isinstance(result, GitHubIssue)
            assert result.id == 2
            assert result.number == 101


@pytest.mark.asyncio
class TestGitHubClientUpdateIssue:
    """Test GitHubClient.update_issue method."""

    @pytest.mark.parametrize("update_data", [
        {"title": "Updated Title"},
        {"body": "Updated body"},
        {"state": "closed"},
        {"labels": ["enhancement"]},
        {"title": "Updated", "body": "New body", "state": "open"},
    ])
    async def test_update_issue_with_various_updates(self, update_data):
        """Test update_issue with various update fields."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(token="ghp_test")

        mock_response = {
            "id": 1,
            "number": 100,
            "title": "Updated Issue",
            "state": "open",
            "user": {"login": "testuser"},
            "assignees": [],
            "labels": [],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-03T00:00:00Z",
            "html_url": "https://github.com/test/repo/issues/100",
        }

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.update_issue("owner", "repo", 100, **update_data)

            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args.kwargs
            assert call_kwargs["method"] == "PATCH"
            assert call_kwargs["endpoint"] == "/repos/owner/repo/issues/100"
            # Only provided fields should be in the data
            for key, value in update_data.items():
                assert call_kwargs["data"].get(key) == value


@pytest.mark.asyncio
class TestGitHubClientCloseIssue:
    """Test GitHubClient.close_issue method."""

    async def test_close_issue_without_comment(self):
        """Test close_issue without comment."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(token="ghp_test")

        mock_response = {
            "id": 1,
            "number": 100,
            "title": "Issue to Close",
            "state": "closed",
            "user": {"login": "testuser"},
            "assignees": [],
            "labels": [],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-03T00:00:00Z",
            "html_url": "https://github.com/test/repo/issues/100",
        }

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.close_issue("owner", "repo", 100)

            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args.kwargs
            assert call_kwargs["method"] == "PATCH"
            assert call_kwargs["endpoint"] == "/repos/owner/repo/issues/100"
            assert call_kwargs["data"]["state"] == "closed"

    async def test_close_issue_with_comment(self):
        """Test close_issue with comment."""
        from app.clients.github_client import GitHubClient

        client = GitHubClient(token="ghp_test")

        mock_response = {
            "id": 1,
            "number": 100,
            "title": "Issue to Close",
            "body": "Closing with comment",
            "state": "closed",
            "user": {"login": "testuser"},
            "assignees": [],
            "labels": [],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-03T00:00:00Z",
            "html_url": "https://github.com/test/repo/issues/100",
        }

        with patch.object(client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.close_issue("owner", "repo", 100, comment="Closing now")

            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args.kwargs
            assert call_kwargs["method"] == "PATCH"
            assert call_kwargs["endpoint"] == "/repos/owner/repo/issues/100"
            assert call_kwargs["data"]["state"] == "closed"
            assert "body" in call_kwargs["data"]
            assert "Closing now" in call_kwargs["data"]["body"]
