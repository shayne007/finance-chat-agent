"""GitHub MCP Server implementation.

This module provides an MCP (Model Context Protocol) server for GitHub operations.
"""

from __future__ import annotations

import logging
from typing import Any

from app.clients.github_client import (
    GitHubAPIError,
    GitHubAuthenticationError,
    GitHubClient,
    GitHubNotFoundError,
    GitHubRateLimitError,
)

logger = logging.getLogger(__name__)


class GitHubMCPServer:
    """MCP Server for GitHub Agent integration.

    This server provides tool discovery and execution for GitHub operations
    including issue listing, creation, updates, and closing.

    Attributes:
        _github_client: GitHubClient instance for API interactions.
    """

    # Tool schemas for GitHub operations
    _LIST_ISSUES_TOOL_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "owner": {
                "type": "string",
                "description": "Repository owner username",
            },
            "repo": {
                "type": "string",
                "description": "Repository name",
            },
            "state": {
                "type": "string",
                "description": "Issue state (open/closed)",
                "enum": ["open", "closed"],
            },
            "assignee": {
                "type": "string",
                "description": "Filter by assignee username",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by labels",
            },
        },
        "required": ["owner", "repo"],
    }

    _CREATE_ISSUE_TOOL_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "owner": {
                "type": "string",
                "description": "Repository owner username",
            },
            "repo": {
                "type": "string",
                "description": "Repository name",
            },
            "title": {
                "type": "string",
                "description": "Issue title",
            },
            "body": {
                "type": "string",
                "description": "Issue body content",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Issue labels",
            },
            "assignees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Usernames to assign",
            },
        },
        "required": ["owner", "repo", "title"],
    }

    _UPDATE_ISSUE_TOOL_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "owner": {
                "type": "string",
                "description": "Repository owner username",
            },
            "repo": {
                "type": "string",
                "description": "Repository name",
            },
            "issue_number": {
                "type": "integer",
                "description": "Issue number",
            },
            "title": {
                "type": "string",
                "description": "New issue title",
            },
            "body": {
                "type": "string",
                "description": "New issue body",
            },
            "state": {
                "type": "string",
                "description": "New issue state",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New issue labels",
            },
            "assignees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New assignees",
            },
        },
        "required": ["owner", "repo", "issue_number"],
    }

    _CLOSE_ISSUE_TOOL_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "owner": {
                "type": "string",
                "description": "Repository owner username",
            },
            "repo": {
                "type": "string",
                "description": "Repository name",
            },
            "issue_number": {
                "type": "integer",
                "description": "Issue number",
            },
            "comment": {
                "type": "string",
                "description": "Comment to add when closing",
            },
        },
        "required": ["owner", "repo", "issue_number"],
    }

    def __init__(self, github_client: GitHubClient) -> None:
        """Initialize the GitHub MCP Server.

        Args:
            github_client: GitHubClient instance for API interactions.
        """
        self._github_client = github_client
        logger.info("GitHubMCPServer initialized")

    @property
    def github_client(self) -> GitHubClient:
        """Get the GitHub client instance."""
        return self._github_client

    async def list_tools(self) -> list[dict[str, Any]]:
        """List all available GitHub tools.

        Returns:
            List of tool definitions with name, description, and input_schema.
        """
        return [
            {
                "name": "github_list_issues",
                "description": "List issues in a GitHub repository with optional filters",
                "input_schema": self._LIST_ISSUES_TOOL_SCHEMA,
            },
            {
                "name": "github_create_issue",
                "description": "Create a new issue in a GitHub repository",
                "input_schema": self._CREATE_ISSUE_TOOL_SCHEMA,
            },
            {
                "name": "github_update_issue",
                "description": "Update an existing issue in a GitHub repository",
                "input_schema": self._UPDATE_ISSUE_TOOL_SCHEMA,
            },
            {
                "name": "github_close_issue",
                "description": "Close an issue in a GitHub repository",
                "input_schema": self._CLOSE_ISSUE_TOOL_SCHEMA,
            },
        ]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a GitHub tool.

        Args:
            tool_name: The name of the tool to execute.
            arguments: Tool-specific arguments.

        Returns:
            Tool execution result with success status and content/error.

        Raises:
            ValueError: If the tool name is unknown.
        """
        logger.info(f"Calling tool: {tool_name} with arguments: {arguments}")

        try:
            if tool_name == "github_list_issues":
                return await self._list_issues(**arguments)
            elif tool_name == "github_create_issue":
                return await self._create_issue(**arguments)
            elif tool_name == "github_update_issue":
                return await self._update_issue(**arguments)
            elif tool_name == "github_close_issue":
                return await self._close_issue(**arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except GitHubAPIError as e:
            logger.error(f"GitHub API error in {tool_name}: {e}")
            return self._error_response(str(e))
        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}: {e}")
            return self._error_response(f"An unexpected error occurred: {e}")

    async def _list_issues(
        self,
        owner: str,
        repo: str,
        state: str | None = None,
        assignee: str | None = None,
        labels: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """List issues in a repository.

        Args:
            owner: Repository owner username.
            repo: Repository name.
            state: Filter by issue state.
            assignee: Filter by assignee.
            labels: Filter by labels.

        Returns:
            Tool execution result with list of issues.
        """
        issues = await self._github_client.list_issues(
            owner=owner,
            repo=repo,
            state=state,
            assignee=assignee,
            labels=labels,
        )
        return {
            "success": True,
            "content": f"Found {len(issues)} issue(s)",
            "result": issues,
        }

    async def _create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new issue.

        Args:
            owner: Repository owner username.
            repo: Repository name.
            title: Issue title.
            body: Issue body.
            labels: Issue labels.
            assignees: Usernames to assign.

        Returns:
            Tool execution result with created issue.
        """
        issue = await self._github_client.create_issue(
            owner=owner,
            repo=repo,
            title=title,
            body=body,
            labels=labels,
            assignees=assignees,
        )
        return {
            "success": True,
            "content": f"Issue #{issue['number']} created successfully",
            "result": issue,
        }

    async def _update_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update an existing issue.

        Args:
            owner: Repository owner username.
            repo: Repository name.
            issue_number: Issue number.
            title: New issue title.
            body: New issue body.
            state: New issue state.
            labels: New issue labels.
            assignees: New assignees.

        Returns:
            Tool execution result with updated issue.
        """
        issue = await self._github_client.update_issue(
            owner=owner,
            repo=repo,
            issue_number=issue_number,
            title=title,
            body=body,
            state=state,
            labels=labels,
            assignees=assignees,
        )
        return {
            "success": True,
            "content": f"Issue #{issue['number']} updated successfully",
            "result": issue,
        }

    async def _close_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        comment: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Close an issue.

        Args:
            owner: Repository owner username.
            repo: Repository name.
            issue_number: Issue number.
            comment: Optional comment to add.

        Returns:
            Tool execution result with closed issue.
        """
        issue = await self._github_client.close_issue(
            owner=owner,
            repo=repo,
            issue_number=issue_number,
            comment=comment,
        )
        return {
            "success": True,
            "content": f"Issue #{issue['number']} closed successfully",
            "result": issue,
        }

    def _error_response(self, error_message: str) -> dict[str, Any]:
        """Create a standardized error response.

        Args:
            error_message: The error message to return.

        Returns:
            Error response dictionary.
        """
        return {
            "success": False,
            "error": error_message,
        }
