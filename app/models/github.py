"""GitHub data models for the GitHub Agent MCP integration.

This module defines Pydantic models for GitHub API responses and agent interactions.
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, computed_field


class GitHubIssue(BaseModel):
    """Represents a GitHub issue.

    Attributes:
        id: The unique identifier for the issue.
        number: The issue number in the repository.
        title: The title of the issue.
        body: The body content of the issue (optional).
        state: The state of the issue (open/closed).
        author: The username of the issue author.
        assignees: List of usernames assigned to the issue.
        labels: List of labels applied to the issue.
        created_at: ISO 8601 timestamp when the issue was created.
        updated_at: ISO 8601 timestamp when the issue was last updated.
        url: The URL to the issue.
    """

    id: int
    number: int
    title: str
    body: Optional[str] = None
    state: str = Field(..., pattern=r'^(open|closed)$')
    author: str
    assignees: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    url: str

    @computed_field
    @property
    def is_open(self) -> bool:
        """Return True if the issue is open."""
        return self.state == "open"

    @computed_field
    @property
    def is_closed(self) -> bool:
        """Return True if the issue is closed."""
        return self.state == "closed"

    @computed_field
    @property
    def has_assignees(self) -> bool:
        """Return True if the issue has assignees."""
        return len(self.assignees) > 0

    @computed_field
    @property
    def has_labels(self) -> bool:
        """Return True if the issue has labels."""
        return len(self.labels) > 0


class GitHubPullRequest(BaseModel):
    """Represents a GitHub pull request.

    Attributes:
        id: The unique identifier for the pull request.
        number: The PR number in the repository.
        title: The title of the pull request.
        body: The body content of the pull request (optional).
        state: The state of the pull request (open/closed/merged).
        author: The username of the PR author.
        head_branch: The name of the source branch.
        base_branch: The name of the target branch.
        created_at: ISO 8601 timestamp when the PR was created.
        url: The URL to the pull request.
    """

    id: int
    number: int
    title: str
    body: Optional[str] = None
    state: str = Field(..., pattern=r'^(open|closed|merged)$')
    author: str
    head_branch: str
    base_branch: str
    created_at: str
    url: str

    @computed_field
    @property
    def is_open(self) -> bool:
        """Return True if the PR is open."""
        return self.state == "open"

    @computed_field
    @property
    def is_closed(self) -> bool:
        """Return True if the PR is closed."""
        return self.state == "closed"

    @computed_field
    @property
    def is_merged(self) -> bool:
        """Return True if the PR is merged."""
        return self.state == "merged"


class GitHubComment(BaseModel):
    """Represents a GitHub comment on an issue or pull request.

    Attributes:
        id: The unique identifier for the comment.
        body: The body content of the comment.
        author: The username of the comment author.
        created_at: ISO 8601 timestamp when the comment was created.
        url: The URL to the comment.
    """

    id: int
    body: str
    author: str
    created_at: str
    url: str


class RepositoryStatus(BaseModel):
    """Represents the status of a repository branch.

    Attributes:
        branch: The name of the branch.
        commit_sha: The SHA of the commit.
        status_checks: List of status check results.
    """

    branch: str
    commit_sha: str
    status_checks: list[dict[str, Any]] = Field(default_factory=list)

    @computed_field
    @property
    def has_status_checks(self) -> bool:
        """Return True if there are status checks."""
        return len(self.status_checks) > 0

    @computed_field
    @property
    def all_checks_passing(self) -> bool:
        """Return True if all status checks are passing."""
        if not self.has_status_checks:
            return True
        return all(check.get("state") == "success" for check in self.status_checks)

    @computed_field
    @property
    def has_failing_checks(self) -> bool:
        """Return True if any status checks are failing."""
        return any(check.get("state") in ("failure", "error") for check in self.status_checks)


class FileContent(BaseModel):
    """Represents the content of a file in a repository.

    Attributes:
        path: The path to the file in the repository.
        content: The file content as a string.
        sha: The SHA of the file blob.
        size: The size of the file in bytes.
    """

    path: str
    content: str
    sha: str
    size: int


class CodeSearchResult(BaseModel):
    """Represents a result from code search.

    Attributes:
        path: The path to the file containing the match.
        score: The relevance score (0.0 to 1.0).
        matches: List of match results with line numbers and snippets.
    """

    path: str
    score: float = Field(..., ge=0.0, le=1.0)
    matches: list[dict[str, Any]] = Field(default_factory=list)


class GitHubIntent(BaseModel):
    """Represents the classified intent of a user query.

    Attributes:
        category: The category of the intent (e.g., issue, pr, search).
        action: The action to perform (e.g., list, create, update).
        confidence: The confidence score (0.0 to 1.0).
        entities: Extracted entities from the query.
    """

    category: str
    action: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    entities: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def is_high_confidence(self) -> bool:
        """Return True if confidence is >= 0.9."""
        return self.confidence >= 0.9

    @computed_field
    @property
    def is_medium_confidence(self) -> bool:
        """Return True if confidence is >= 0.7 and < 0.9."""
        return 0.7 <= self.confidence < 0.9

    @computed_field
    @property
    def is_low_confidence(self) -> bool:
        """Return True if confidence is < 0.7."""
        return self.confidence < 0.7


class AgentResponse(BaseModel):
    """Represents the response from the GitHub Agent.

    Attributes:
        success: Whether the operation was successful.
        content: The main response content.
        metadata: Additional metadata about the response.
        error: Error message if the operation failed.
    """

    success: bool
    content: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
