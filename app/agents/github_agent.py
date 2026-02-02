"""GitHub Agent for processing GitHub-related queries.

This agent provides intent classification and routing for GitHub operations
using an LLM and MCP server for tool execution.
"""

from __future__ import annotations

import logging
from typing import Any

from app.mcp.github_server import GitHubMCPServer
from app.models.github import GitHubIntent

logger = logging.getLogger(__name__)


class GitHubAgent:
    """Agent for processing GitHub-related queries.

    This agent uses an LLM for intent classification and routes
    queries to the appropriate MCP server tools.

    Attributes:
        llm: Language model for intent classification.
        mcp_server: MCP server for tool execution.
        github_client: GitHub client for API interactions.
        default_repo: Default repository to use if not specified in query.
        _intent_parser: Pydantic model for parsing LLM responses.
    """

    INTENT_SYSTEM_PROMPT: str = """You are a GitHub assistant that classifies user queries.

    Your task is to classify the user's intent into a category and action.

Categories:
- issue: Operations related to GitHub issues (list, create, update, close)
- pr: Operations related to pull requests (list, create, merge, close)
- repo: Repository-wide operations (status, branches, files)
- search: Code or content search operations
- unknown: Anything else

Actions for issues:
- list: Show/list issues
- create: Create a new issue
- update: Update an existing issue
- close: Close an issue

Actions for PRs:
- list: Show/list pull requests
- create: Create a new pull request
- merge: Merge a pull request
- close: Close a pull request

Actions for repo:
- status: Show repository status
- branches: List branches
- files: Show file contents

Actions for search:
- code: Search code in the repository

Extract entities from the query:
- repo: Repository name (e.g., "owner/repo")
- issue_number: Issue number (e.g., "#123")
- title: Issue title (for create/update)
- state: Issue state (open/closed)
- labels: Issue labels
- assignee: Username to assign

Return a JSON object with this structure:
{
    "category": "issue|pr|repo|search|unknown",
    "action": "list|create|update|close|merge|status|branches|files|code",
    "confidence": 0.0 to 1.0,
    "entities": {
        "repo": "extracted repo or null",
        "issue_number": 123 or null,
        "title": "extracted title or null",
        "state": "open|closed or null",
        "labels": ["label1", "label2"] or null,
        "assignee": "username or null"
    }
}

If you're unsure about the category or action, set confidence to a lower value.
"""

    def __init__(
        self,
        llm: Any,
        mcp_server: GitHubMCPServer,
        github_client: Any,
        default_repo: str | None = None,
    ) -> None:
        """Initialize the GitHub Agent.

        Args:
            llm: Language model for intent classification.
            mcp_server: MCP server for tool execution.
            github_client: GitHub client for API interactions.
            default_repo: Default repository to use.
        """
        self.llm = llm
        self.mcp_server = mcp_server
        self.github_client = github_client
        self.default_repo = default_repo
        self._intent_parser = GitHubIntent

        logger.info("GitHubAgent initialized")

    async def classify_intent(self, query: str) -> GitHubIntent:
        """Classify the user's query into a GitHub intent.

        Args:
            query: The user's natural language query.

        Returns:
            GitHubIntent with category, action, and confidence.
        """
        logger.debug(f"Classifying intent for query: {query}")

        try:
            response = await self.llm.ainvoke(
                self.INTENT_SYSTEM_PROMPT + "\n\nQuery: " + query
            )

            # Parse the response
            intent_data = self._parse_intent_response(response)

            return GitHubIntent(**intent_data)

        except Exception as e:
            logger.exception(f"Intent classification failed: {e}")
            # Return low-confidence fallback
            return GitHubIntent(
                category="unknown",
                action="unknown",
                confidence=0.3,
                entities={},
            )

    def _parse_intent_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM response to extract intent data.

        Args:
            response: The LLM's text response.

        Returns:
            Parsed intent data as a dictionary.
        """
        # This is a simplified parser - in production you'd want
        # more robust JSON parsing with error handling
        import json

        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Fallback: parse using heuristics
                return self._parse_intent_heuristically(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM response, using fallback")
            return self._parse_intent_heuristically(response)

    def _parse_intent_heuristically(self, response: str) -> dict[str, Any]:
        """Parse intent using heuristics when JSON parsing fails.

        Args:
            response: The LLM's text response.

        Returns:
            Parsed intent data as a dictionary.
        """
        response_lower = response.lower()

        # Default values
        category = "unknown"
        action = "unknown"
        confidence = 0.3
        entities = {}

        # Determine category
        if any(word in response_lower for word in ["issue", "issues", "bug", "ticket"]):
            category = "issue"
            if any(word in response_lower for word in ["list", "show", "find", "search"]):
                action = "list"
            elif any(word in response_lower for word in ["create", "new", "add"]):
                action = "create"
            elif any(word in response_lower for word in ["close", "resolve"]):
                action = "close"
            elif any(word in response_lower for word in ["update", "edit", "change"]):
                action = "update"
        elif any(word in response_lower for word in ["pr", "pull", "request", "merge"]):
            category = "pr"
        elif any(word in response_lower for word in ["status", "branch", "file"]):
            category = "repo"
        elif any(word in response_lower for word in ["search", "find", "look"]):
            category = "search"
            action = "code"

        return {
            "category": category,
            "action": action,
            "confidence": confidence,
            "entities": entities,
        }

    async def process_query(self, query: str) -> dict[str, Any]:
        """Process a user query for GitHub operations.

        Args:
            query: The user's natural language query.

        Returns:
            AgentResponse with success status and content/error.
        """
        logger.info(f"Processing query: {query}")

        try:
            # Classify the intent
            intent = await self.classify_intent(query)

            # Extract repository from entities or use default
            repo = intent.entities.get("repo") or self.default_repo

            if not repo:
                return self._error_response(
                    "No repository specified. Please specify a repository "
                    "(e.g., 'list issues in owner/repo') or set a default repo."
                )

            # Route to the appropriate tool
            tool_name, tool_args = self._route_to_tool(
                intent.category, intent.action, intent.entities, repo
            )

            # Execute the tool
            result = await self.mcp_server.call_tool(tool_name, tool_args)

            # Format the response
            content = self._format_response(intent.action, result)

            return {
                "success": True,
                "content": content,
                "metadata": {
                    "tool": tool_name,
                    "intent_category": intent.category,
                    "intent_action": intent.action,
                    "confidence": intent.confidence,
                    "repo": repo,
                },
            }

        except Exception as e:
            logger.exception(f"Query processing failed: {e}")
            return self._error_response(f"An error occurred while processing your query: {e}")

    def _route_to_tool(
        self,
        category: str,
        action: str,
        entities: dict[str, Any],
        repo: str,
    ) -> tuple[str, dict[str, Any]]:
        """Route an intent to the appropriate MCP tool.

        Args:
            category: The intent category.
            action: The intent action.
            entities: Extracted entities from the query.
            repo: The repository to operate on.

        Returns:
            Tuple of (tool_name, tool_args).

        Raises:
            ValueError: If the intent cannot be routed.
        """
        # Parse repository if specified in entities
        if "repo" in entities:
            owner_repo = entities["repo"]
            if "/" in owner_repo:
                owner, repo_name = owner_repo.split("/", 1)
            else:
                # Assume it's just the repo name, use owner from elsewhere
                # For now, we'll default to a simple split
                parts = owner_repo.split("/")
                if len(parts) == 2:
                    owner, repo_name = parts
                else:
                    owner = repo_name = "unknown"
        else:
            owner, repo_name = repo.split("/", 1)

        tool_args: dict[str, Any] = {"owner": owner, "repo": repo_name}

        if category == "issue":
            tool_args["state"] = entities.get("state")
            tool_args["assignee"] = entities.get("assignee")
            if entities.get("labels"):
                tool_args["labels"] = entities["labels"]

            if action == "list":
                return "github_list_issues", tool_args
            elif action == "create":
                tool_args["title"] = entities.get("title", "")
                tool_args["body"] = entities.get("body")
                tool_args["labels"] = entities.get("labels")
                tool_args["assignees"] = entities.get("assignees")
                return "github_create_issue", tool_args
            elif action == "update":
                tool_args["issue_number"] = entities.get("issue_number")
                tool_args["title"] = entities.get("title")
                tool_args["body"] = entities.get("body")
                tool_args["state"] = entities.get("state")
                return "github_update_issue", tool_args
            elif action == "close":
                tool_args["issue_number"] = entities.get("issue_number")
                tool_args["comment"] = entities.get("comment")
                return "github_close_issue", tool_args
            else:
                # Default to list for issue category
                return "github_list_issues", tool_args

        elif category == "pr":
            # PR operations would go here
            return "github_list_issues", tool_args  # Placeholder

        elif category == "repo":
            # Repo operations would go here
            return "github_list_issues", tool_args  # Placeholder

        elif category == "search":
            # Search operations would go here
            return "github_list_issues", tool_args  # Placeholder

        else:
            # Unknown category - try list issues as a fallback
            return "github_list_issues", tool_args

    def _format_response(
        self, action: str, tool_result: dict[str, Any]
    ) -> str:
        """Format the tool result into a user-friendly response.

        Args:
            action: The action that was executed.
            tool_result: The result from the MCP server.

        Returns:
            User-friendly response string.
        """
        if not tool_result.get("success", False):
            return f"Error: {tool_result.get('error', 'Unknown error')}"

        result_data = tool_result.get("result")
        content = tool_result.get("content", "")

        if action == "list":
            if result_data and len(result_data) > 0:
                count = len(result_data)
                if count == 1:
                    return f"{content} Here are the details:"
                else:
                    return f"{content} Showing the first 10 of {count}:"
                for i, issue in enumerate(result_data[:10]):
                    return f"- #{issue['number']}: {issue['title']} ({issue['state']})"
                return f"No issues found in the repository."
            else:
                return content

        elif action == "create":
            return content

        elif action == "update":
            return content

        elif action == "close":
            return content

        else:
            return f"Action '{action}' completed. {content}"

    def _error_response(self, error_message: str) -> dict[str, Any]:
        """Create an error response.

        Args:
            error_message: The error message.

        Returns:
            Error response dictionary.
        """
        return {
            "success": False,
            "error": error_message,
            "metadata": {
                "error_type": "github_agent_error",
            },
        }
