"""Tests for GitHub MCP Server using TDD approach."""

import pytest
from unittest.mock import AsyncMock, Mock, patch


@pytest.mark.asyncio
class TestGitHubMCPServerInitialization:
    """Test GitHubMCPServer initialization."""

    async def test_server_initializes_with_github_client(self):
        """Test GitHubMCPServer initializes with GitHubClient dependency."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)
        server = GitHubMCPServer(github_client=mock_github_client)

        assert server.github_client is mock_github_client

    async def test_server_initializes_without_errors(self):
        """Test GitHubMCPServer starts without errors."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)
        server = GitHubMCPServer(github_client=mock_github_client)

        # Should not raise any exceptions during initialization
        assert server is not None
        assert isinstance(server, GitHubMCPServer)

    async def test_server_stores_github_client_as_instance_variable(self):
        """Test GitHubMCPServer stores GitHubClient as instance variable."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)
        server = GitHubMCPServer(github_client=mock_github_client)

        assert hasattr(server, "_github_client")
        assert server._github_client is mock_github_client


@pytest.mark.asyncio
class TestGitHubMCPServerListTools:
    """Test GitHubMCPServer list_tools method."""

    async def test_list_tools_returns_all_github_tools(self):
        """Test list_tools returns all GitHub tools."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)
        server = GitHubMCPServer(github_client=mock_github_client)

        tools = await server.list_tools()

        assert len(tools) == 4
        tool_names = [tool["name"] for tool in tools]
        assert "github_list_issues" in tool_names
        assert "github_create_issue" in tool_names
        assert "github_update_issue" in tool_names
        assert "github_close_issue" in tool_names

    async def test_list_tools_returns_tool_schemas_with_required_fields(self):
        """Test list_tools returns tool schemas with required fields."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)
        server = GitHubMCPServer(github_client=mock_github_client)

        tools = await server.list_tools()

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool


@pytest.mark.asyncio
class TestGitHubMCPServerToolSchemas:
    """Test GitHubMCPServer tool schemas."""

    async def test_github_list_issues_tool_schema(self):
        """Test github_list_issues tool schema (name, description, input_schema)."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)
        server = GitHubMCPServer(github_client=mock_github_client)

        tools = await server.list_tools()
        list_issues_tool = next(
            (t for t in tools if t["name"] == "github_list_issues"), None
        )

        assert list_issues_tool is not None
        assert list_issues_tool["name"] == "github_list_issues"
        assert "description" in list_issues_tool
        assert "input_schema" in list_issues_tool

    async def test_github_create_issue_tool_schema(self):
        """Test github_create_issue tool schema."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)
        server = GitHubMCPServer(github_client=mock_github_client)

        tools = await server.list_tools()
        create_issue_tool = next(
            (t for t in tools if t["name"] == "github_create_issue"), None
        )

        assert create_issue_tool is not None
        assert create_issue_tool["name"] == "github_create_issue"
        assert "title" in create_issue_tool["input_schema"]

    async def test_github_update_issue_tool_schema(self):
        """Test github_update_issue tool schema."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)
        server = GitHubMCPServer(github_client=mock_github_client)

        tools = await server.list_tools()
        update_issue_tool = next(
            (t for t in tools if t["name"] == "github_update_issue"), None
        )

        assert update_issue_tool is not None
        assert update_issue_tool["name"] == "github_update_issue"
        assert "issue_number" in update_issue_tool["input_schema"]

    async def test_github_close_issue_tool_schema(self):
        """Test github_close_issue tool schema."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)
        server = GitHubMCPServer(github_client=mock_github_client)

        tools = await server.list_tools()
        close_issue_tool = next(
            (t for t in tools if t["name"] == "github_close_issue"), None
        )

        assert close_issue_tool is not None
        assert close_issue_tool["name"] == "github_close_issue"
        assert "issue_number" in close_issue_tool["input_schema"]


@pytest.mark.asyncio
class TestGitHubMCPServerCallTool:
    """Test GitHubMCPServer call_tool method."""

    @pytest.mark.parametrize("tool_name,arguments", [
        ("github_list_issues", {"owner": "test", "repo": "test-repo"}),
        ("github_create_issue", {"owner": "test", "repo": "test-repo", "title": "New Issue"}),
        ("github_update_issue", {"owner": "test", "repo": "test-repo", "issue_number": 1}),
        ("github_close_issue", {"owner": "test", "repo": "test-repo", "issue_number": 1}),
    ])
    async def test_call_tool_with_valid_arguments(self, tool_name, arguments):
        """Test call_tool with valid arguments for each tool."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)

        # Mock the GitHubClient methods
        if tool_name == "github_list_issues":
            mock_github_client.list_issues = AsyncMock(return_value=[])
        elif tool_name == "github_create_issue":
            mock_github_client.create_issue = AsyncMock(return_value={"id": 1})
        elif tool_name == "github_update_issue":
            mock_github_client.update_issue = AsyncMock(return_value={"id": 1})
        elif tool_name == "github_close_issue":
            mock_github_client.close_issue = AsyncMock(return_value={"id": 1})

        server = GitHubMCPServer(github_client=mock_github_client)

        result = await server.call_tool(tool_name, arguments)

        assert result is not None
        assert "success" in result or "content" in result or "result" in result

    async def test_call_tool_with_unknown_tool_name(self):
        """Test call_tool with unknown tool name raises error."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)
        server = GitHubMCPServer(github_client=mock_github_client)

        with pytest.raises(ValueError) as exc_info:
            await server.call_tool("unknown_tool", {})

        assert "unknown" in str(exc_info.value).lower()


@pytest.mark.asyncio
class TestGitHubMCPServerErrorHandling:
    """Test GitHubMCPServer error handling."""

    async def test_call_tool_transforms_github_exceptions_to_mcp_errors(self):
        """Test call_tool transforms GitHub exceptions to MCP errors."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient, GitHubAPIError

        mock_github_client = Mock(spec=GitHubClient)
        mock_github_client.list_issues = AsyncMock(
            side_effect=GitHubAPIError("Test error")
        )

        server = GitHubMCPServer(github_client=mock_github_client)

        result = await server.call_tool("github_list_issues", {"owner": "test", "repo": "test"})

        assert result["success"] is False
        assert "error" in result

    async def test_error_messages_are_user_friendly(self):
        """Test error messages are user-friendly."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient, GitHubNotFoundError

        mock_github_client = Mock(spec=GitHubClient)
        mock_github_client.list_issues = AsyncMock(
            side_effect=GitHubNotFoundError("Repository not found")
        )

        server = GitHubMCPServer(github_client=mock_github_client)

        result = await server.call_tool("github_list_issues", {"owner": "test", "repo": "test"})

        assert result["success"] is False
        assert result["error"] is not None
        assert len(result["error"]) > 0
