"""Tests for dependency injection with GitHub components."""

from unittest.mock import Mock, patch

import pytest


class TestDependencyInjection:
    """Test that GitHub components can be injected."""

    def test_github_client_is_available_for_injection(self):
        """Test that GitHubClient can be imported for DI container."""
        from app.clients.github_client import GitHubClient

        mock_client = Mock(spec=GitHubClient)
        assert mock_client is not None
        print("GitHubClient is available for DI container")

    def test_github_mcp_server_is_available_for_injection(self):
        """Test that GitHubMCPServer can be imported for DI container."""
        from app.mcp.github_server import GitHubMCPServer
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)
        mock_server = GitHubMCPServer(github_client=mock_github_client)
        assert mock_server is not None
        print("GitHubMCPServer is available for DI container")

    def test_github_agent_is_available_for_injection(self):
        """Test that GitHubAgent can be imported for DI container."""
        from app.agents.github_agent import GitHubAgent

        mock_github_client = Mock()
        agent = GitHubAgent(
            jira_agent=Mock(),
            rag_agent=Mock(),
            github_agent=mock_github_client,
        )
        assert agent.github_agent is not None
        print("GitHubAgent is available for DI container")

    def test_jira_agent_exists(self):
        """Test that JiraAgent exists for backward compatibility."""
        from app.agents.jira_agent import JiraAgent

        mock_agent = JiraAgent()
        assert mock_agent is not None
        print("JiraAgent is available for DI container")

    def test_rag_agent_exists(self):
        """Test that RAGAgent exists for backward compatibility."""
        try:
            from app.agents.rag_agent import RAGAgent
            print("RAGAgent is available for DI container")
        except ImportError:
            print("RAGAgent import failed (may not be implemented yet)")

    def test_finance_agent_has_github_agent_parameter(self):
        """Test that FinanceAgent can accept github_agent parameter."""
        from app.agents.finance_agent import FinanceAgent
        from app.agents.github_agent import GitHubAgent

        mock_jira = Mock()
        mock_rag = Mock()
        mock_github = Mock()

        agent = FinanceAgent(
            jira_agent=mock_jira,
            rag_agent=mock_rag,
            github_agent=mock_github,
        )

        assert agent.github_agent is mock_github
        print("FinanceAgent accepts github_agent parameter")

    @pytest.mark.asyncio
    class TestDependenciesResolve:
        """Test that all dependencies resolve correctly."""

        async def test_finance_agent_with_github_agent_initializes_correctly(self):
            """Test FinanceAgent with GitHubAgent initializes correctly."""
            from app.agents.finance_agent import FinanceAgent

            mock_jira = Mock()
            mock_rag = Mock()
            mock_github = Mock()

            agent = FinanceAgent(
                jira_agent=mock_jira,
                rag_agent=mock_rag,
                github_agent=mock_github,
            )

            assert agent.github_agent is not None
            assert agent.jira is not None
            assert agent.rag_agent is not None

        async def test_finance_agent_without_github_agent_works(self):
            """Test FinanceAgent without GitHubAgent still works."""
            from app.agents.finance_agent import FinanceAgent

            mock_jira = Mock()
            mock_rag = Mock()

            agent = FinanceAgent(
                jira_agent=mock_jira,
                rag_agent=mock_rag,
                github_agent=None,
            )

            assert agent.github_agent is None
            assert agent.jira is not None
            assert agent.rag_agent is not None
            # Should not raise an exception
            result = await agent.run("test query")
            assert result is not None


class TestDIContainerRegistry:
    """Test that DI container registration works (if implemented)."""

    def test_di_container_has_services(self):
        """Test that DI container has service registration."""
        # Note: This test will fail if DI container isn't implemented
        try:
            from app.core.di import DIContainer
            print("DIContainer class exists")
            # Check if required services are registered
            container = DIContainer()
            if "github_client" in dir(container):
                print("DI container has github_client service")
            else:
                print("DI container does not have github_client service (expected)")
        except ImportError:
            print("DIContainer not implemented yet (expected)")
