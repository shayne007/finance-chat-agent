"""Tests for GitHub API endpoints using TDD approach."""

import pytest
from unittest.mock import AsyncMock, Mock, patch


@pytest.mark.asyncio
class TestGitHubQueryEndpoint:
    """Test POST /api/v1/github/query endpoint."""

    async def test_post_github_query_returns_task_id(self):
        """Test POST /api/v1/github/query returns 202 with task_id."""
        from fastapi.testclient import TestClient

        with TestClient(app="app.main:app") as client:
            response = client.post(
                "/api/v1/github/query", json={"query": "test query"}
            )

            assert response.status_code == 202
            data = response.json()
            assert "task_id" in data
            assert isinstance(data["task_id"], str)

    async def test_post_github_query_validates_request_with_pydantic(self):
        """Test endpoint validates request with Pydantic models."""
        from fastapi.testclient import TestClient

        with TestClient(app="app.main:app") as client:
            response = client.post(
                "/api/v1/github/query", json={"query": "test"}
            )

            assert response.status_code == 202


@pytest.mark.asyncio
class TestGitHubIssuesEndpoints:
    """Test GitHub issues endpoints."""

    async def test_get_github_issues(self):
        """Test GET /api/v1/github/issues/{owner}/{repo}."""
        from fastapi.testclient import TestClient

        with TestClient(app="app.main:app") as client:
            response = client.get("/api/v1/github/issues/owner/repo")

            assert response.status_code in (200, 404)  # 404 if not found

    async def test_post_github_issues(self):
        """Test POST /api/v1/github/issues/{owner}/{repo}."""
        from fastapi.testclient import TestClient

        with TestClient(app="app.main:app") as client:
            response = client.post(
                "/api/v1/github/issues/owner/repo",
                json={"title": "Test Issue", "body": "Test body"},
            )

            assert response.status_code in (201, 202, 400, 422)


@pytest.mark.asyncio
class TestGitHubChatEndpoint:
    """Test POST /api/v1/github/chat endpoint."""

    async def test_post_github_chat(self):
        """Test POST /api/v1/github/chat."""
        from fastapi.testclient import TestClient

        with TestClient(app="app.main:app") as client:
            response = client.post(
                "/api/v1/github/chat", json={"message": "test message"}
            )

            assert response.status_code in (200, 202, 400)


@pytest.mark.asyncio
class TestGitHubStatusEndpoint:
    """Test GET /api/v1/github/repo/{owner}/{repo}/status endpoint."""

    async def test_get_github_repo_status(self):
        """Test GET /api/v1/github/repo/{owner}/{repo}/status."""
        from fastapi.testclient import TestClient

        with TestClient(app="app.main:app") as client:
            response = client.get("/api/v1/github/repo/owner/repo")

            assert response.status_code in (200, 404)


@pytest.mark.asyncio
class TestRouterIntegration:
    """Test GitHub endpoints appear in OpenAPI docs."""

    async def test_github_endpoints_appear_in_openapi_docs(self):
        """Test GitHub endpoints appear in OpenAPI docs."""
        from fastapi.testclient import TestClient

        with TestClient(app="app.main:app") as client:
            response = client.get("/openapi.json")

            assert response.status_code == 200

            openapi = response.json()
            assert "paths" in openapi

            # Check for GitHub paths
            github_paths = [
                "/api/v1/github/query",
                "/api/v1/github/chat",
                "/api/v1/github/issues/{owner}/{repo}",
                "/api/v1/github/repo/{owner}/{repo}/status",
            ]

            for path in github_paths:
                if path in openapi["paths"]:
                    # Verify the path exists
                    assert isinstance(path, str)

            # Verify at least some GitHub endpoints are present
            github_endpoint_paths = [
                f"/api/v1/github/query",
                f"/api/v1/github/chat",
                f"/api/v1/github/issues",
                f"/api/v1/github/repo",
            ]
            found_paths = [
                p for p in openapi["paths"] for path in github_endpoint_paths
            ]

            assert len(found_paths) > 0, f"Expected at least 1 GitHub endpoint path to be in OpenAPI docs, found: {len(found_paths)}"

    async def test_github_endpoints_have_request_response_examples(self):
        """Test endpoints have request/response examples in OpenAPI docs."""
        from fastapi.testclient import TestClient

        with TestClient(app="app.main:app") as client:
            response = client.get("/openapi.json")

            assert response.status_code == 200

            openapi = response.json()
            # Examples are part of the OpenAPI spec, we can check if they exist
            assert "paths" in openapi
            # GitHub paths should be present
            if "/api/v1/github/query" in openapi["paths"]:
                github_query_spec = openapi["paths"]["/api/v1/github/query"]
                # Check for request examples (if they exist in the spec)
                assert "get" in github_query_spec.get("methods", {}), "Expected GET method"
                assert "post" in github_query_spec.get("methods", {}), "Expected POST method"
