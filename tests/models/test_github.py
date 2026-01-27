"""Tests for GitHub models using TDD approach."""

import pytest
from datetime import datetime
from pydantic import ValidationError


class TestGitHubIssue:
    """Test GitHubIssue Pydantic model validation."""

    @pytest.mark.parametrize("data,expected_valid", [
        (
            {
                "id": 1,
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "author": "testuser",
                "assignees": ["user1", "user2"],
                "labels": ["bug", "enhancement"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            True
        ),
        (
            {
                "id": 2,
                "number": 101,
                "title": "Another Issue",
                "body": None,
                "state": "closed",
                "author": "anotheruser",
                "assignees": [],
                "labels": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/101"
            },
            True
        ),
    ])
    def test_github_issue_valid_data(self, data, expected_valid):
        """Test GitHubIssue accepts valid data."""
        from app.models.github import GitHubIssue

        if expected_valid:
            issue = GitHubIssue(**data)
            assert issue.id == data["id"]
            assert issue.number == data["number"]
            assert issue.title == data["title"]
            assert issue.state == data["state"]

    @pytest.mark.parametrize("data,missing_field", [
        (
            {
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "author": "testuser",
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "id"
        ),
        (
            {
                "id": 1,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "author": "testuser",
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "number"
        ),
        (
            {
                "id": 1,
                "number": 100,
                "body": "Test body",
                "state": "open",
                "author": "testuser",
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "title"
        ),
        (
            {
                "id": 1,
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "author": "testuser",
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "state"
        ),
        (
            {
                "id": 1,
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "author"
        ),
        (
            {
                "id": 1,
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "author": "testuser",
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "updated_at"
        ),
        (
            {
                "id": 1,
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "author": "testuser",
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z"
            },
            "url"
        ),
    ])
    def test_github_issue_missing_required_fields(self, data, missing_field):
        """Test GitHubIssue raises ValidationError for missing required fields."""
        from app.models.github import GitHubIssue

        with pytest.raises(ValidationError) as exc_info:
            GitHubIssue(**data)
        assert missing_field in str(exc_info.value).lower() or missing_field in [f['loc'][0] for f in exc_info.value.errors()]

    @pytest.mark.parametrize("data,field,wrong_value", [
        (
            {
                "id": "not_an_int",
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "author": "testuser",
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "id",
            "not_an_int"
        ),
        (
            {
                "id": 1,
                "number": "not_an_int",
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "author": "testuser",
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "number",
            "not_an_int"
        ),
        (
            {
                "id": 1,
                "number": 100,
                "title": 123,
                "body": "Test body",
                "state": "open",
                "author": "testuser",
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "title",
            123
        ),
        (
            {
                "id": 1,
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "state": "invalid_state",
                "author": "testuser",
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "state",
            "invalid_state"
        ),
        (
            {
                "id": 1,
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "author": ["not_a_string"],
                "assignees": ["user1"],
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "author",
            ["not_a_string"]
        ),
        (
            {
                "id": 1,
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "author": "testuser",
                "assignees": "not_a_list",
                "labels": ["bug"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "assignees",
            "not_a_list"
        ),
        (
            {
                "id": 1,
                "number": 100,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "author": "testuser",
                "assignees": ["user1"],
                "labels": 123,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100"
            },
            "labels",
            123
        ),
    ])
    def test_github_issue_invalid_types(self, data, field, wrong_value):
        """Test GitHubIssue raises ValidationError for wrong types."""
        from app.models.github import GitHubIssue

        with pytest.raises(ValidationError):
            GitHubIssue(**data)


class TestGitHubPullRequest:
    """Test GitHubPullRequest Pydantic model validation."""

    @pytest.mark.parametrize("data,expected_valid", [
        (
            {
                "id": 1,
                "number": 10,
                "title": "Test PR",
                "body": "Test PR body",
                "state": "open",
                "author": "testuser",
                "head_branch": "feature-branch",
                "base_branch": "main",
                "created_at": "2024-01-01T00:00:00Z",
                "url": "https://github.com/owner/repo/pull/10"
            },
            True
        ),
        (
            {
                "id": 2,
                "number": 11,
                "title": "Another PR",
                "body": None,
                "state": "closed",
                "author": "anotheruser",
                "head_branch": "develop",
                "base_branch": "main",
                "created_at": "2024-01-01T00:00:00Z",
                "url": "https://github.com/owner/repo/pull/11"
            },
            True
        ),
    ])
    def test_github_pull_request_valid_data(self, data, expected_valid):
        """Test GitHubPullRequest accepts valid data."""
        from app.models.github import GitHubPullRequest

        if expected_valid:
            pr = GitHubPullRequest(**data)
            assert pr.id == data["id"]
            assert pr.number == data["number"]
            assert pr.title == data["title"]
            assert pr.state == data["state"]
            assert pr.head_branch == data["head_branch"]
            assert pr.base_branch == data["base_branch"]


class TestGitHubComment:
    """Test GitHubComment Pydantic model validation."""

    @pytest.mark.parametrize("data,expected_valid", [
        (
            {
                "id": 1,
                "body": "Test comment",
                "author": "testuser",
                "created_at": "2024-01-01T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/100#comment-1"
            },
            True
        ),
        (
            {
                "id": 2,
                "body": "Another comment",
                "author": "anotheruser",
                "created_at": "2024-01-01T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/101#comment-2"
            },
            True
        ),
    ])
    def test_github_comment_valid_data(self, data, expected_valid):
        """Test GitHubComment accepts valid data."""
        from app.models.github import GitHubComment

        if expected_valid:
            comment = GitHubComment(**data)
            assert comment.id == data["id"]
            assert comment.body == data["body"]
            assert comment.author == data["author"]


class TestRepositoryStatus:
    """Test RepositoryStatus Pydantic model validation."""

    @pytest.mark.parametrize("data,expected_valid", [
        (
            {
                "branch": "main",
                "commit_sha": "abc123",
                "status_checks": [
                    {"name": "ci", "state": "success"},
                    {"name": "tests", "state": "pending"}
                ]
            },
            True
        ),
        (
            {
                "branch": "develop",
                "commit_sha": "def456",
                "status_checks": []
            },
            True
        ),
    ])
    def test_repository_status_valid_data(self, data, expected_valid):
        """Test RepositoryStatus accepts valid data."""
        from app.models.github import RepositoryStatus

        if expected_valid:
            status = RepositoryStatus(**data)
            assert status.branch == data["branch"]
            assert status.commit_sha == data["commit_sha"]
            assert len(status.status_checks) == len(data["status_checks"])


class TestFileContent:
    """Test FileContent Pydantic model validation."""

    @pytest.mark.parametrize("data,expected_valid", [
        (
            {
                "path": "src/main.py",
                "content": "print('hello')",
                "sha": "abc123",
                "size": 100
            },
            True
        ),
        (
            {
                "path": "README.md",
                "content": "# Project",
                "sha": "def456",
                "size": 50
            },
            True
        ),
    ])
    def test_file_content_valid_data(self, data, expected_valid):
        """Test FileContent accepts valid data."""
        from app.models.github import FileContent

        if expected_valid:
            file_content = FileContent(**data)
            assert file_content.path == data["path"]
            assert file_content.content == data["content"]
            assert file_content.sha == data["sha"]
            assert file_content.size == data["size"]


class TestCodeSearchResult:
    """Test CodeSearchResult Pydantic model validation."""

    @pytest.mark.parametrize("data,expected_valid", [
        (
            {
                "path": "src/main.py",
                "score": 0.95,
                "matches": [
                    {"line": 10, "snippet": "def main():"},
                    {"line": 15, "snippet": "return result"}
                ]
            },
            True
        ),
        (
            {
                "path": "utils.py",
                "score": 0.80,
                "matches": []
            },
            True
        ),
    ])
    def test_code_search_result_valid_data(self, data, expected_valid):
        """Test CodeSearchResult accepts valid data."""
        from app.models.github import CodeSearchResult

        if expected_valid:
            result = CodeSearchResult(**data)
            assert result.path == data["path"]
            assert result.score == data["score"]
            assert len(result.matches) == len(data["matches"])


class TestGitHubIntent:
    """Test GitHubIntent Pydantic model validation."""

    @pytest.mark.parametrize("data,expected_valid", [
        (
            {
                "category": "issue",
                "action": "list",
                "confidence": 0.95,
                "entities": {
                    "repo": "owner/repo",
                    "state": "open"
                }
            },
            True
        ),
        (
            {
                "category": "pr",
                "action": "create",
                "confidence": 0.88,
                "entities": {
                    "repo": "owner/repo",
                    "title": "New PR"
                }
            },
            True
        ),
    ])
    def test_github_intent_valid_data(self, data, expected_valid):
        """Test GitHubIntent accepts valid data."""
        from app.models.github import GitHubIntent

        if expected_valid:
            intent = GitHubIntent(**data)
            assert intent.category == data["category"]
            assert intent.action == data["action"]
            assert intent.confidence == data["confidence"]
            assert intent.entities == data["entities"]


class TestAgentResponse:
    """Test AgentResponse Pydantic model validation."""

    @pytest.mark.parametrize("data,expected_valid", [
        (
            {
                "success": True,
                "content": "Found 3 issues",
                "metadata": {"count": 3, "tool": "list_issues"},
                "error": None
            },
            True
        ),
        (
            {
                "success": False,
                "content": None,
                "metadata": {"tool": "list_issues"},
                "error": "Authentication failed"
            },
            True
        ),
        (
            {
                "success": True,
                "content": "Issue created successfully",
                "metadata": {"issue_number": 100}
            },
            True
        ),
    ])
    def test_agent_response_valid_data(self, data, expected_valid):
        """Test AgentResponse accepts valid data."""
        from app.models.github import AgentResponse

        if expected_valid:
            response = AgentResponse(**data)
            assert response.success == data["success"]
            assert response.content == data["content"]
            assert response.error == data.get("error")
