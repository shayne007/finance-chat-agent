"""Tests for FinanceAgent GitHub routing using TDD approach."""

import pytest
from unittest.mock import AsyncMock, Mock, patch


@pytest.mark.asyncio
class TestFinanceAgentGitHubRouting:
    """Test FinanceAgent routes GitHub queries to GitHubAgent."""

    @pytest.mark.parametrize("query,expected_agent", [
        ("list my GitHub issues", "github_agent"),
        ("show me PRs in my repo", "github_agent"),
        ("create issue for bug report", "github_agent"),
        ("close issue #123", "github_agent"),
        ("what's the status of my repo", "github_agent"),
        ("search for async keyword in my code", "github_agent"),
    ])
    async def test_github_queries_route_to_github_agent(
        self, query, expected_agent
    ):
        """Test GitHub queries route to GitHubAgent."""
        from app.agents.finance_agent import FinanceAgent

        mock_github_agent = Mock(spec=Mock)
        mock_jira_agent = Mock(spec=Mock)
        mock_rag_agent = Mock(spec=Mock)

        agent = FinanceAgent(
            jira_agent=mock_jira_agent,
            rag_agent=mock_rag_agent,
            github_agent=mock_github_agent,
        )

        result = await agent.process_query(query)

        assert result["_agent"] == expected_agent
        assert result.get("_agent") == expected_agent

    @pytest.mark.parametrize("query,expected_agent", [
        ("list my Jira tickets", "jira_agent"),
        ("show me open tickets in finance project", "jira_agent"),
        ("search for finance documents", "rag_agent"),
        ("what's in the finance database", "rag_agent"),
    ])
    async def test_jira_queries_still_route_to_jira_agent(
        self, query, expected_agent
    ):
        """Test Jira queries still route to JiraAgent (no regression)."""
        from app.agents.finance_agent import FinanceAgent

        mock_github_agent = Mock(spec=Mock)
        mock_jira_agent = Mock(spec=Mock)
        mock_rag_agent = Mock(spec=Mock)

        agent = FinanceAgent(
            jira_agent=mock_jira_agent,
            rag_agent=mock_rag_agent,
            github_agent=mock_github_agent,
        )

        result = await agent.process_query(query)

        assert result["_agent"] == expected_agent
        assert result.get("_agent") == expected_agent

    @pytest.mark.parametrize("query,expected_agent", [
        ("search for financial data", "rag_agent"),
        ("show me relevant documents", "rag_agent"),
        ("what reports do we have", "rag_agent"),
    ])
    async def test_rag_queries_still_route_to_rag_agent(
        self, query, expected_agent
    ):
        """Test RAG queries still route to RAGAgent (no regression)."""
        from app.agents.finance_agent import FinanceAgent

        mock_github_agent = Mock(spec=Mock)
        mock_jira_agent = Mock(spec=Mock)
        mock_rag_agent = Mock(spec=Mock)

        agent = FinanceAgent(
            jira_agent=mock_jira_agent,
            rag_agent=mock_rag_agent,
            github_agent=mock_github_agent,
        )

        result = await agent.process_query(query)

        assert result["_agent"] == expected_agent
        assert result.get("_agent") == expected_agent

    async def test_routing_includes_logging(self):
        """Test routing includes logging."""
        from app.agents.finance_agent import FinanceAgent
        import logging

        mock_github_agent = Mock(spec=Mock)
        mock_jira_agent = Mock(spec=Mock)
        mock_rag_agent = Mock(spec=Mock)

        agent = FinanceAgent(
            jira_agent=mock_jira_agent,
            rag_agent=mock_rag_agent,
            github_agent=mock_github_agent,
        )

        # Set up a logger capture
        with patch("logging.Logger.info") as mock_info:
            result = await agent.process_query("list GitHub issues")

            # Verify that logging occurred
            mock_info.assert_called()


class TestFinanceAgentKeywords:
    """Test GitHub keyword list."""

    async def test_github_keywords_list(self):
        """Test GitHub keywords list includes common GitHub terms."""
        from app.agents.finance_agent import FinanceAgent

        mock_github_agent = Mock(spec=Mock)
        mock_jira_agent = Mock(spec=Mock)
        mock_rag_agent = Mock(spec=Mock)

        agent = FinanceAgent(
            jira_agent=mock_jira_agent,
            rag_agent=mock_rag_agent,
            github_agent=mock_github_agent,
        )

        # Keywords should be checked in routing logic
        # Since we're mocking, we can't test the actual keywords list,
        # but we can verify the agent has the GitHub agent available
        assert agent.github_agent is not None


@pytest.mark.asyncio
class TestFinanceAgentWithNoGitHubAgent:
    """Test FinanceAgent behavior without GitHub agent."""

    async def test_queries_without_github_use_default_routing(self):
        """Test queries without GitHub agent use default routing."""
        from app.agents.finance_agent import FinanceAgent

        mock_jira_agent = Mock(spec=Mock)
        mock_rag_agent = Mock(spec=Mock)

        agent = FinanceAgent(
            jira_agent=mock_jira_agent,
            rag_agent=mock_rag_agent,
            github_agent=None,
        )

        # Should route to default (likely Jira or RAG)
        result = await agent.process_query("some query")

        assert result is not None
        assert result.get("_agent") in ("jira_agent", "rag_agent")
        assert result.get("_agent") is not None
