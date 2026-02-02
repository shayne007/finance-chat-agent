"""Tests for GitHub Agent using TDD approach."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.clients.github_client import GitHubAPIError


@pytest.mark.asyncio
class TestGitHubAgentInitialization:
    """Test GitHubAgent initialization."""

    async def test_agent_initializes_with_llm_and_mcp_server(self):
        """Test GitHubAgent initializes with LLM and MCP server."""
        from app.agents.github_agent import GitHubAgent
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_llm = Mock()
        mock_mcp_server = Mock(spec=GitHubMCPServer)
        mock_github_client = Mock(spec=GitHubClient)

        agent = GitHubAgent(
            llm=mock_llm,
            mcp_server=mock_mcp_server,
            github_client=mock_github_client,
        )

        assert agent.llm is mock_llm
        assert agent.mcp_server is mock_mcp_server
        assert agent.github_client is mock_github_client

    @pytest.mark.parametrize("default_repo", [
        "owner/repo",
        None,
    ])
    async def test_agent_initializes_with_default_repo_parameter(
        self, default_repo
    ):
        """Test GitHubAgent initializes with default_repo parameter."""
        from app.agents.github_agent import GitHubAgent
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_llm = Mock()
        mock_mcp_server = Mock(spec=GitHubMCPServer)
        mock_github_client = Mock(spec=GitHubClient)

        agent = GitHubAgent(
            llm=mock_llm,
            mcp_server=mock_mcp_server,
            github_client=mock_github_client,
            default_repo=default_repo,
        )

        assert agent.default_repo == default_repo

    async def test_agent_initializes_pydantic_output_parser(self):
        """Test GitHubAgent initializes Pydantic output parser."""
        from app.agents.github_agent import GitHubAgent
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient
        from app.models.github import GitHubIntent

        mock_llm = Mock()
        mock_mcp_server = Mock(spec=GitHubMCPServer)
        mock_github_client = Mock(spec=GitHubClient)

        agent = GitHubAgent(
            llm=mock_llm,
            mcp_server=mock_mcp_server,
            github_client=mock_github_client,
        )

        # Check that the parser exists
        assert hasattr(agent, "_intent_parser")
        assert isinstance(agent._intent_parser, type)


@pytest.mark.asyncio
class TestGitHubAgentClassifyIntent:
    """Test GitHubAgent classify_intent method."""

    @pytest.mark.parametrize("query,expected_category,expected_action", [
        ("list my issues", "issue", "list"),
        ("create issue about bug", "issue", "create"),
        ("close issue #123", "issue", "close"),
        ("update issue 456", "issue", "update"),
        ("show open issues in my repo", "issue", "list"),
    ])
    async def test_classify_intent_with_various_queries(
        self, query, expected_category, expected_action
    ):
        """Test classify_intent with various queries."""
        from app.agents.github_agent import GitHubAgent
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient
        from app.models.github import GitHubIntent

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={
                "category": expected_category,
                "action": expected_action,
                "confidence": 0.95,
                "entities": {},
            }
        )

        mock_mcp_server = Mock(spec=GitHubMCPServer)
        mock_github_client = Mock(spec=GitHubClient)

        agent = GitHubAgent(
            llm=mock_llm,
            mcp_server=mock_mcp_server,
            github_client=mock_github_client,
        )

        result = await agent.classify_intent(query)

        assert isinstance(result, GitHubIntent)
        assert result.category == expected_category
        assert result.action == expected_action
        assert result.confidence >= 0.7

    async def test_classify_intent_handles_parsing_errors(self):
        """Test classify_intent handles parsing errors gracefully."""
        from app.agents.github_agent import GitHubAgent
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("Parse error"))

        mock_mcp_server = Mock(spec=GitHubMCPServer)
        mock_github_client = Mock(spec=GitHubClient)

        agent = GitHubAgent(
            llm=mock_llm,
            mcp_server=mock_mcp_server,
            github_client=mock_github_client,
        )

        # Should not raise an exception, but handle it gracefully
        result = await agent.classify_intent("test query")

        # Should return a low-confidence fallback intent
        assert result.confidence < 0.5


@pytest.mark.asyncio
class TestGitHubAgentProcessQuery:
    """Test GitHubAgent process_query method."""

    @pytest.mark.parametrize("query,expected_tool", [
        ("list issues", "github_list_issues"),
        ("show my issues", "github_list_issues"),
        ("create a new issue", "github_create_issue"),
        ("close issue 123", "github_close_issue"),
    ])
    async def test_process_query_with_valid_queries(self, query, expected_tool):
        """Test process_query with valid GitHub queries."""
        from app.agents.github_agent import GitHubAgent
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient
        from app.models.github import AgentResponse

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={
                "category": "issue",
                "action": "list",
                "confidence": 0.95,
                "entities": {},
            }
        )

        mock_mcp_server = Mock(spec=GitHubMCPServer)
        mock_mcp_server.call_tool = AsyncMock(
            return_value={
                "success": True,
                "content": "Found 5 issues",
                "result": [],
            }
        )

        mock_github_client = Mock(spec=GitHubClient)

        agent = GitHubAgent(
            llm=mock_llm,
            mcp_server=mock_mcp_server,
            github_client=mock_github_client,
            default_repo="owner/repo",
        )

        result = await agent.process_query(query)

        assert isinstance(result, AgentResponse)
        assert result.success is True

    @pytest.mark.parametrize("query,error_message", [
        ("list issues", None),  # Default repo should work
        ("show my issues", None),  # Default repo should work
    ])
    async def test_process_query_with_default_repo(self, query, error_message):
        """Test process_query uses default repo when not in context."""
        from app.agents.github_agent import GitHubAgent
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            return_value={
                "category": "issue",
                "action": "list",
                "confidence": 0.95,
                "entities": {},
            }
        )

        mock_mcp_server = Mock(spec=GitHubMCPServer)
        mock_mcp_server.call_tool = AsyncMock(
            return_value={
                "success": True,
                "content": "Found 5 issues",
                "result": [],
            }
        )

        mock_github_client = Mock(spec=GitHubClient)

        agent = GitHubAgent(
            llm=mock_llm,
            mcp_server=mock_mcp_server,
            github_client=mock_github_client,
            default_repo="owner/repo",
        )

        result = await agent.process_query(query)

        assert result.success is True


@pytest.mark.asyncio
class TestGitHubAgentResponseFormatting:
    """Test GitHubAgent response formatting."""

    @pytest.mark.parametrize("action,results,expected_content", [
        (
            "list",
            [],
            "No issues found in the repository.",
        ),
        (
            "list",
            [
                {
                    "id": 1,
                    "number": 100,
                    "title": "Issue #100",
                    "state": "open",
                    "author": "user",
                    "assignees": [],
                    "labels": [],
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-02T00:00:00Z",
                    "url": "https://github.com/test/repo/issues/100",
                }
            ],
            "Found 1 issue(s) in the repository.",
        ),
    ])
    async def test_format_response_with_various_actions(
        self, action, results, expected_content
    ):
        """Test _format_response with various actions and results."""
        from app.agents.github_agent import GitHubAgent
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_llm = Mock()
        mock_mcp_server = Mock(spec=GitHubMCPServer)
        mock_github_client = Mock(spec=GitHubClient)

        agent = GitHubAgent(
            llm=mock_llm,
            mcp_server=mock_mcp_server,
            github_client=mock_github_client,
        )

        result = agent._format_response(action, results)

        assert expected_content in result
        # Check response is conversational and user-friendly
        assert isinstance(result, str)


@pytest.mark.asyncio
class TestGitHubAgentErrorHandling:
    """Test GitHubAgent error handling."""

    @pytest.mark.parametrize("error_type", [
        "intent_classification",
        "tool_execution",
        "context_extraction",
    ])
    async def test_process_query_with_errors_returns_agent_response_with_failure(
        self, error_type
    ):
        """Test process_query with errors returns AgentResponse with success=False."""
        from app.agents.github_agent import GitHubAgent
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient, GitHubAPIError
        from app.models.github import AgentResponse

        mock_llm = AsyncMock()

        if error_type == "intent_classification":
            mock_llm.ainvoke = AsyncMock(
                side_effect=Exception("LLM error")
            )
            mock_mcp_server = Mock(spec=GitHubMCPServer)
        elif error_type == "tool_execution":
            mock_llm.ainvoke = AsyncMock(
                return_value={
                    "category": "issue",
                    "action": "list",
                    "confidence": 0.95,
                    "entities": {},
                }
            )
            mock_mcp_server = Mock(spec=GitHubMCPServer)
            mock_mcp_server.call_tool = AsyncMock(
                side_effect=GitHubAPIError("API error")
            )
        else:  # context_extraction
            mock_llm.ainvoke = AsyncMock(
                return_value={
                    "category": "unknown",
                    "action": "unknown",
                    "confidence": 0.3,
                    "entities": {},
                }
            )
            mock_mcp_server = Mock(spec=GitHubMCPServer)

        mock_github_client = Mock(spec=GitHubClient)

        agent = GitHubAgent(
            llm=mock_llm,
            mcp_server=mock_mcp_server,
            github_client=mock_github_client,
        )

        # Process query should not raise, but return error response
        result = await agent.process_query("test query")

        assert isinstance(result, AgentResponse)
        assert result.success is False
        assert result.error is not None

    async def test_errors_have_user_friendly_messages(self):
        """Test errors have user-friendly messages."""
        from app.agents.github_agent import GitHubAgent
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient, GitHubAPIError

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(
            side_effect=GitHubAPIError("Test error")
        )

        mock_mcp_server = Mock(spec=GitHubMCPServer)
        mock_github_client = Mock(spec=GitHubClient)

        agent = GitHubAgent(
            llm=mock_llm,
            mcp_server=mock_mcp_server,
            github_client=mock_github_client,
        )

        result = await agent.process_query("test query")

        assert result.success is False
        assert result.error is not None
        assert len(result.error) > 0
        # Error message should be user-friendly, not technical
        assert "stack trace" not in result.error.lower()

    @pytest.mark.parametrize("error_type,side_effect", [
        ("rate_limit", GitHubAPIError("Rate limit exceeded")),
        ("auth_error", GitHubAPIError("Authentication failed")),
        ("not_found", GitHubAPIError("Resource not found")),
    ])
    async def test_errors_include_details_in_metadata(
        self, error_type, side_effect
    ):
        """Test errors include details in metadata."""
        from app.agents.github_agent import GitHubAgent
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient, GitHubAPIError

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=side_effect)

        mock_mcp_server = Mock(spec=GitHubMCPServer)
        mock_github_client = Mock(spec=GitHubClient)

        agent = GitHubAgent(
            llm=mock_llm,
            mcp_server=mock_mcp_server,
            github_client=mock_github_client,
        )

        result = await agent.process_query("test query")

        assert result.success is False
        assert result.metadata is not None
        assert "tool" in result.metadata or "error_type" in result.metadata
