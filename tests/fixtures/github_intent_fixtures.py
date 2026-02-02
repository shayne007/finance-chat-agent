"""GitHub intent classification test fixtures.

This module provides example queries with their expected classifications
for testing the GitHub agent's intent classification capabilities.
"""

GITHUB_INTENT_FIXTURES: list[dict[str, str | None]] = [
    # Issue list queries
    {
        "query": "list my issues",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.9,
        "description": "Simple list query",
    },
    {
        "query": "show me all issues",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.9,
        "description": "Show all query",
    },
    {
        "query": "list issues in my repo",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.8,
        "description": "List in repo query",
    },
    {
        "query": "what issues do I have",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.8,
        "description": "What issues query",
    },
    {
        "query": "find my open issues",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.9,
        "description": "Find with filter query",
    },
    {
        "query": "show me closed issues",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.9,
        "description": "Show with state filter",
    },
    {
        "query": "list bugs and enhancements",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.85,
        "description": "List with labels query",
    },
    {
        "query": "show issues assigned to me",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.85,
        "description": "Assigned to query",
    },
    # Issue create queries
    {
        "query": "create a new issue",
        "expected_category": "issue",
        "expected_action": "create",
        "expected_confidence_min": 0.9,
        "description": "Simple create query",
    },
    {
        "query": "file a bug report",
        "expected_category": "issue",
        "expected_action": "create",
        "expected_confidence_min": 0.9,
        "description": "File bug query",
    },
    {
        "query": "I need to report an issue",
        "expected_category": "issue",
        "expected_action": "create",
        "expected_confidence_min": 0.9,
        "description": "Report issue query",
    },
    {
        "query": "create issue about database connection",
        "expected_category": "issue",
        "expected_action": "create",
        "expected_confidence_min": 0.85,
        "description": "Create with context query",
    },
    {
        "query": "add a new bug to the backlog",
        "expected_category": "issue",
        "expected_action": "create",
        "expected_confidence_min": 0.85,
        "description": "Add to backlog query",
    },
    {
        "query": "create an issue with labels",
        "expected_category": "issue",
        "expected_action": "create",
        "expected_confidence_min": 0.8,
        "description": "Create with labels query",
    },
    # Issue update queries
    {
        "query": "update issue #123",
        "expected_category": "issue",
        "expected_action": "update",
        "expected_confidence_min": 0.95,
        "description": "Update with number query",
    },
    {
        "query": "change the title of issue 456",
        "expected_category": "issue",
        "expected_action": "update",
        "expected_confidence_min": 0.9,
        "description": "Change title query",
    },
    {
        "query": "add a comment to issue #789",
        "expected_category": "issue",
        "expected_action": "update",
        "expected_confidence_min": 0.9,
        "description": "Add comment query",
    },
    {
        "query": "reassign this issue to John",
        "expected_category": "issue",
        "expected_action": "update",
        "expected_confidence_min": 0.9,
        "description": "Reassign query",
    },
    # Issue close queries
    {
        "query": "close issue #123",
        "expected_category": "issue",
        "expected_action": "close",
        "expected_confidence_min": 0.95,
        "description": "Close with number query",
    },
    {
        "query": "resolve this issue",
        "expected_category": "issue",
        "expected_action": "close",
        "expected_confidence_min": 0.9,
        "description": "Resolve synonym query",
    },
    {
        "query": "this issue is fixed, close it",
        "expected_category": "issue",
        "expected_action": "close",
        "expected_confidence_min": 0.9,
        "description": "Fixed close query",
    },
    {
        "query": "mark as resolved and add comment",
        "expected_category": "issue",
        "expected_action": "close",
        "expected_confidence_min": 0.85,
        "description": "Resolve with comment query",
    },
    # Pull request queries
    {
        "query": "show my pull requests",
        "expected_category": "pr",
        "expected_action": "list",
        "expected_confidence_min": 0.9,
        "description": "Show PRs query",
    },
    {
        "query": "list open PRs",
        "expected_category": "pr",
        "expected_action": "list",
        "expected_confidence_min": 0.9,
        "description": "List open PRs",
    },
    {
        "query": "create a pull request",
        "expected_category": "pr",
        "expected_action": "create",
        "expected_confidence_min": 0.9,
        "description": "Create PR query",
    },
    {
        "query": "merge my PR",
        "expected_category": "pr",
        "expected_action": "merge",
        "expected_confidence_min": 0.9,
        "description": "Merge PR query",
    },
    # Repository queries
    {
        "query": "show repository status",
        "expected_category": "repo",
        "expected_action": "status",
        "expected_confidence_min": 0.9,
        "description": "Show status query",
    },
    {
        "query": "what branches are available",
        "expected_category": "repo",
        "expected_action": "branches",
        "expected_confidence_min": 0.9,
        "description": "Branches query",
    },
    {
        "query": "list all files in src",
        "expected_category": "repo",
        "expected_action": "files",
        "expected_confidence_min": 0.9,
        "description": "List files query",
    },
    # Search queries
    {
        "query": "search for 'async' keyword",
        "expected_category": "search",
        "expected_action": "code",
        "expected_confidence_min": 0.9,
        "description": "Search keyword query",
    },
    {
        "query": "find code that uses X library",
        "expected_category": "search",
        "expected_action": "code",
        "expected_confidence_min": 0.85,
        "description": "Find code query",
    },
    {
        "query": "search in utils folder",
        "expected_category": "search",
        "expected_action": "code",
        "expected_confidence_min": 0.85,
        "description": "Search folder query",
    },
    # Repository specific queries
    {
        "query": "list issues in facebook/react",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.9,
        "description": "External repo query",
    },
    {
        "query": "what issues are open in shayne007/finance-chat-agent",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.9,
        "description": "Specific repo query",
    },
    {
        "query": "create issue in myrepo for bug",
        "expected_category": "issue",
        "expected_action": "create",
        "expected_confidence_min": 0.8,
        "description": "Contextual create query",
    },
    # Mixed/complex queries
    {
        "query": "show me my open bugs",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.9,
        "description": "Filtered list query",
    },
    {
        "query": "find issues assigned to john about database",
        "expected_category": "issue",
        "expected_action": "list",
        "expected_confidence_min": 0.85,
        "description": "Filtered with assignee and topic",
    },
    {
        "query": "I need to report a bug in the authentication module",
        "expected_category": "issue",
        "expected_action": "create",
        "expected_confidence_min": 0.9,
        "description": "Detailed create query",
    },
    # Vague queries
    {
        "query": "help me with GitHub",
        "expected_category": "unknown",
        "expected_action": "unknown",
        "expected_confidence_min": 0.3,
        "description": "Vague help query",
    },
    {
        "query": "I have a question",
        "expected_category": "unknown",
        "expected_action": "unknown",
        "expected_confidence_min": 0.3,
        "description": "Vague question query",
    },
    {
        "query": "can you check something",
        "expected_category": "unknown",
        "expected_action": "unknown",
        "expected_confidence_min": 0.3,
        "description": "Vague check query",
    },
]


def get_accuracy_score(results: list[dict[str, Any]]) -> float:
    """Calculate accuracy score for intent classification.

    Args:
        results: List of classification results with actual and expected.

    Returns:
            Accuracy score between 0.0 and 1.0.
    """
    if not results:
        return 0.0

    correct = 0
    total = len(results)

    for result in results:
        actual_category = result.get("actual_category")
        expected_category = result.get("expected_category")
        actual_action = result.get("actual_action")
        expected_action = result.get("expected_action")

        if (
            actual_category == expected_category
            and actual_action == expected_action
        ):
            correct += 1

    return correct / total if total > 0 else 0.0
