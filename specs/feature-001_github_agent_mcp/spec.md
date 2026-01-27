# Spec: GitHub Chat Agent with MCP Integration

**Feature ID:** feature-001_github_agent_mcp
**Status:** Draft
**Version:** 1.0
**Last Updated:** 2026-01-27

---

## 1. Executive Summary

### 1.1 Problem Statement
Users of the finance chat agent currently need to switch contexts to GitHub to manage repository-related tasks (issues, PRs, code reviews, repository status). This context switching disrupts workflow efficiency and creates a fragmented user experience.

### 1.2 Solution
Implement a **GitHub Agent** using the **Model Context Protocol (MCP)** to enable seamless GitHub operations directly within the chat interface. This agent will integrate with existing FinanceAgent orchestration to provide a unified conversational experience.

### 1.3 Success Criteria
- ✅ Users can query GitHub issues and PRs through natural language
- ✅ Users can create, update, and close GitHub issues via chat
- ✅ Users can review and comment on PRs conversationally
- ✅ Integration follows MCP standards for tool composability
- ✅ 95%+ test coverage with integration tests
- ✅ Response time < 5 seconds for GitHub API operations
- ✅ Zero breaking changes to existing agents

---

## 2. User Stories

### 2.1 Primary Users

| Role | Needs | Pain Points |
|------|-------|-------------|
| **Developer** | Quick issue status, PR reviews, branch info | Context switching to GitHub UI |
| **Product Manager** | Issue triage, milestone tracking, release status | Multiple tool navigation |
| **Tech Lead** | Code review oversight, repository metrics | Scattered information sources |

### 2.2 User Story Details

**US-001: Query Issues**
> As a developer, I want to ask "What issues are assigned to me?" so that I can quickly understand my workload without opening GitHub.

**US-002: Create Issue**
> As a product manager, I want to say "Create an issue for the payment bug" so that I can capture requirements during conversation.

**US-003: PR Review**
> As a tech lead, I want to request "Show me open PRs in the backend repo" so that I can review pending changes efficiently.

**US-004: Repository Status**
> As a developer, I want to ask "What's the status of the main branch?" so that I can check deployment readiness.

---

## 3. Functional Requirements

### 3.1 Core Capabilities

#### FR-001: Issue Management
| Capability | Description | Priority |
|------------|-------------|----------|
| List issues | Query issues with filters (assignee, label, state, milestone) | P0 |
| Create issue | Create new issue with title, description, labels, assignees | P0 |
| Update issue | Modify issue state, labels, assignees | P1 |
| Close issue | Close issues with optional comment | P1 |

#### FR-002: Pull Request Management
| Capability | Description | Priority |
|------------|-------------|----------|
| List PRs | Query PRs with filters (state, author, reviewer, branch) | P0 |
| View PR details | Display PR diff, files changed, discussion | P1 |
| Create PR comment | Add review comments to PR | P2 |
| Merge PR | Merge PR with specified method | P2 |

#### FR-003: Repository Operations
| Capability | Description | Priority |
|------------|-------------|----------|
| Repo status | Show branch status, recent commits | P1 |
| File lookup | Retrieve file contents or metadata | P2 |
| Search code | Search code across repository | P2 |

#### FR-004: MCP Tool Integration
| Capability | Description | Priority |
|------------|-------------|----------|
| MCP server | Expose GitHub operations as MCP tools | P0 |
| Tool discovery | Advertise available GitHub capabilities | P0 |
| Context injection | Include repo context in agent prompts | P1 |

### 3.2 Intent Classification

The agent must classify user intents into categories:

```
github:issue:list      - Query/list issues
github:issue:create    - Create new issue
github:issue:update    - Modify existing issue
github:issue:close     - Close an issue
github:pr:list         - Query/list pull requests
github:pr:view         - View PR details
github:pr:comment      - Comment on PR
github:repo:status     - Repository/branch status
github:file:get        - Retrieve file contents
github:code:search     - Search code
github:general         - General GitHub questions
```

### 3.3 Non-Functional Requirements

| Requirement | Specification |
|-------------|---------------|
| **Performance** | GitHub API calls < 5s for 95th percentile |
| **Reliability** | 99.5% uptime for GitHub agent operations |
| **Rate Limiting** | Respect GitHub API rate limits (5000/hr authenticated) |
| **Security** | Store GitHub tokens securely, never expose in logs |
| **Extensibility** | New GitHub features addable without core changes |
| **Testing** | 95%+ code coverage, integration tests for all GitHub API calls |
| **Documentation** | All public APIs documented with docstrings |

---

## 4. Technical Architecture

### 4.1 System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / Client                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI App                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  Chat API     │  │  GitHub API   │  │  MCP Server   │       │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘       │
│          │                  │                  │               │
│  ┌───────▼──────────────────▼──────────────────▼───────┐       │
│  │              FinanceAgent (LangGraph)                │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │       │
│  │  │ JiraAgent   │  │ RAGAgent    │  │GitHubAgent  │ │       │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │       │
│  └────────────────────────────────────────────────────┘       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        External Services                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   GitHub    │  │    Jira     │  │   OpenAI    │             │
│  │     API     │  │     API     │  │     API     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Design

#### 4.2.1 GitHubAgent (`app/agents/github_agent.py`)

**Responsibilities:**
- Intent classification for GitHub-related queries
- Route to appropriate GitHub MCP tools
- Format responses from GitHub API for conversational display
- Handle GitHub API errors gracefully

**Interface:**
```python
class GitHubAgent:
    """Agent for handling GitHub operations via MCP."""

    def __init__(self, github_token: str, default_repo: str):
        """Initialize with GitHub credentials and default repository."""
        ...

    async def process_query(self, query: str, context: dict) -> AgentResponse:
        """Process a GitHub-related query and return response."""
        ...

    def classify_intent(self, query: str) -> GitHubIntent:
        """Classify the intent of a GitHub query."""
        ...
```

#### 4.2.2 GitHub MCP Server (`app/mcp/github_server.py`)

**Responsibilities:**
- Implement MCP server protocol
- Expose GitHub operations as MCP tools
- Handle tool execution and context injection
- Manage tool discovery and capabilities

**MCP Tools Specification:**

| Tool Name | Description | Input Schema | Output Schema |
|-----------|-------------|--------------|---------------|
| `list_issues` | Query GitHub issues | `{owner, repo, state, assignee, labels}` | `{issues: [{id, title, state, assignee, labels}]}` |
| `create_issue` | Create a new issue | `{owner, repo, title, body, labels, assignees}` | `{issue: {id, url, number}}` |
| `update_issue` | Update existing issue | `{owner, repo, issue_number, updates}` | `{issue: {id, updated_fields}}` |
| `list_prs` | Query pull requests | `{owner, repo, state, author, head_branch}` | `{prs: [{id, title, state, author, head}]}` |
| `get_pr_details` | Get PR details and diff | `{owner, repo, pr_number}` | `{pr: {id, title, files_changed, diff}}` |
| `create_pr_comment` | Comment on a PR | `{owner, repo, pr_number, body, path, position}` | `{comment: {id, url}}` |
| `get_repo_status` | Get repository/branch status | `{owner, repo, branch}` | `{branch, commit, status_checks}` |
| `get_file_contents` | Retrieve file contents | `{owner, repo, path, ref}` | `{content, sha, size}` |
| `search_code` | Search code across repo | `{owner, repo, query}` | `{results: [{path, score, matches}]}` |

**Interface:**
```python
class GitHubMCPServer:
    """MCP Server exposing GitHub API as tools."""

    async def start(self):
        """Start the MCP server."""
        ...

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a GitHub MCP tool."""
        ...

    async def list_tools(self) -> list[Tool]:
        """List available MCP tools."""
        ...
```

#### 4.2.3 GitHub Client (`app/clients/github_client.py`)

**Responsibilities:**
- Wrap GitHub REST API v3
- Handle authentication and rate limiting
- Implement retry logic with exponential backoff
- Parse and validate GitHub API responses

**Interface:**
```python
class GitHubClient:
    """Client for GitHub REST API v3."""

    def __init__(self, token: str, base_url: str = "https://api.github.com"):
        """Initialize with GitHub personal access token."""
        ...

    async def list_issues(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        **filters
    ) -> list[Issue]:
        """List issues with optional filters."""
        ...

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> Issue:
        """Create a new issue."""
        ...
    # ... additional GitHub API methods
```

### 4.3 MCP Protocol Integration

The implementation will follow the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) specification:

1. **Server Implementation**: GitHubMCPServer implements MCP server
2. **Tool Registration**: GitHub operations exposed as MCP tools
3. **Context Injection**: Repository context injected into agent prompts
4. **Tool Discovery**: Clients can discover available GitHub capabilities

### 4.4 Data Models

```python
# app/models/github.py
from pydantic import BaseModel
from typing import Literal, Optional

class GitHubIssue(BaseModel):
    """GitHub issue model."""
    id: int
    number: int
    title: str
    body: Optional[str]
    state: Literal["open", "closed"]
    author: Optional[str]
    assignees: list[str]
    labels: list[str]
    created_at: str
    updated_at: str
    url: str

class GitHubPullRequest(BaseModel):
    """GitHub pull request model."""
    id: int
    number: int
    title: str
    body: Optional[str]
    state: Literal["open", "closed", "merged"]
    author: str
    head_branch: str
    base_branch: str
    created_at: str
    url: str

class GitHubIntent(BaseModel):
    """Intent classification result for GitHub queries."""
    category: str
    action: str
    confidence: float
    entities: dict
```

---

## 5. Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal**: Core GitHub client and basic issue operations

| Task | Deliverable | Owner |
|------|-------------|-------|
| 1.1 | Create `GitHubClient` with authentication | - |
| 1.2 | Implement `list_issues` and `create_issue` | - |
| 1.3 | Add GitHub models in `app/models/github.py` | - |
| 1.4 | Write unit tests for GitHub client | - |
| 1.5 | Add GitHub configuration to `app/core/config.py` | - |

**Exit Criteria:**
- Can list and create issues via Python client
- 80%+ test coverage for implemented methods
- Configuration supports GitHub token and default repo

### Phase 2: MCP Server (Week 2-3)
**Goal**: MCP server exposing GitHub tools

| Task | Deliverable | Owner |
|------|-------------|-------|
| 2.1 | Create `GitHubMCPServer` skeleton | - |
| 2.2 | Implement MCP protocol handlers | - |
| 2.3 | Expose GitHub operations as MCP tools | - |
| 2.4 | Implement tool discovery endpoint | - |
| 2.5 | Add integration tests for MCP server | - |

**Exit Criteria:**
- MCP server starts and advertises GitHub tools
- Can call tools via MCP protocol
- Integration tests pass for all exposed tools

### Phase 3: GitHub Agent (Week 3-4)
**Goal**: Conversational agent for GitHub operations

| Task | Deliverable | Owner |
|------|-------------|-------|
| 3.1 | Create `GitHubAgent` with intent classification | - |
| 3.2 | Implement LLM-based intent detection | - |
| 3.3 | Add response formatting for GitHub data | - |
| 3.4 | Implement error handling and retries | - |
| 3.5 | Add agent tests with mock GitHub responses | - |

**Exit Criteria:**
- Agent correctly classifies GitHub intents
- Handles error cases gracefully
- Test coverage > 90%

### Phase 4: FinanceAgent Integration (Week 4-5)
**Goal**: Integrate GitHubAgent into existing orchestration

| Task | Deliverable | Owner |
|------|-------------|-------|
| 4.1 | Add GitHub route to FinanceAgent LangGraph | - |
| 4.2 | Update message routing logic | - |
| 4.3 | Add GitHub-specific API endpoints | - |
| 4.4 | Update Celery tasks for GitHub processing | - |
| 4.5 | Integration tests for end-to-end flow | - |

**Exit Criteria:**
- Users can query GitHub through chat API
- GitHub queries route correctly to GitHubAgent
- No breaking changes to existing agents

### Phase 5: Advanced Features (Week 5-6)
**Goal**: PR operations, search, and repository status

| Task | Deliverable | Owner |
|------|-------------|-------|
| 5.1 | Implement PR list and details | - |
| 5.2 | Add PR comment creation | - |
| 5.3 | Implement code search | - |
| 5.4 | Add repository status endpoint | - |
| 5.5 | Tests for all new features | - |

**Exit Criteria:**
- All P0 and P1 features implemented
- Full feature test suite passing
- Documentation complete

### Phase 6: Documentation & Hardening (Week 6-7)
**Goal**: Production readiness

| Task | Deliverable | Owner |
|------|-------------|-------|
| 6.1 | Write API documentation | - |
| 6.2 | Add user guide for GitHub commands | - |
| 6.3 | Performance testing and optimization | - |
| 6.4 | Security audit of token handling | - |
| 6.5 | Load testing with concurrent requests | - |

**Exit Criteria:**
- Documentation published
- Security review passed
- Performance benchmarks met
- Ready for production deployment

---

## 6. Testing Strategy

### 6.1 Testing Pyramid

```
           ┌──────────────────┐
           │   E2E Tests      │  10%
           │  (integration)   │
           ├──────────────────┤
           │  Integration     │  30%
           │  (w/ GitHub API) │
           ├──────────────────┤
           │   Unit Tests     │  60%
           │  (mocked)        │
           └──────────────────┘
```

### 6.2 Test Coverage Requirements

| Component | Target Coverage | Critical Tests |
|-----------|-----------------|----------------|
| `GitHubClient` | 90%+ | API calls, error handling, rate limiting |
| `GitHubAgent` | 90%+ | Intent classification, routing |
| `GitHubMCPServer` | 85%+ | Tool execution, protocol compliance |
| Integration | 80%+ | End-to-end flows with mock GitHub |

### 6.3 Test Categories

#### Unit Tests
- Mock GitHub API responses using `responses` library
- Test all client methods with various response codes
- Test intent classification with example queries
- Test error handling and retry logic

#### Integration Tests
- Use GitHub Test API or test repositories
- Test real API calls with test tokens
- Test MCP tool execution end-to-end
- Test agent routing and response formatting

#### E2E Tests
- Test complete flow from chat API to GitHub
- Test Celery task processing for GitHub queries
- Test concurrent request handling
- Test rate limiting behavior

### 6.4 Example Test Cases

```python
# tests/agents/test_github_agent.py

import pytest
from app.agents.github_agent import GitHubAgent, GitHubIntent

@pytest.mark.parametrize("query,expected_intent", [
    ("What issues are assigned to me?", GitHubIntent(category="issue", action="list")),
    ("Create an issue for the login bug", GitHubIntent(category="issue", action="create")),
    ("Show me open PRs in backend", GitHubIntent(category="pr", action="list")),
    ("What's the status of main?", GitHubIntent(category="repo", action="status")),
])
def test_intent_classification(agent, query, expected_intent):
    """Test GitHub query intent classification."""
    intent = agent.classify_intent(query)
    assert intent.category == expected_intent.category
    assert intent.action == expected_intent.action
    assert intent.confidence > 0.7

@pytest.mark.asyncio
async def test_process_issue_list_query(agent, mock_github_client):
    """Test processing of issue list query."""
    mock_github_client.list_issues.return_value = [
        GitHubIssue(id=1, title="Bug in login", state="open", ...)
    ]

    response = await agent.process_query(
        "What issues are open?",
        context={"repo": "test/repo"}
    )

    assert "issues" in response.content.lower()
    assert response.success is True
```

---

## 7. Configuration

### 7.1 Environment Variables

```bash
# GitHub Configuration
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
GITHUB_DEFAULT_REPO=owner/repository
GITHUB_BASE_URL=https://api.github.com

# MCP Configuration
MCP_GITHUB_SERVER_ENABLED=true
MCP_GITHUB_SERVER_PORT=3000
MCP_GITHUB_MAX_TOOLS=20

# Agent Configuration
GITHUB_AGENT_ENABLED=true
GITHUB_AGENT_MODEL=gpt-4
GITHUB_AGENT_TEMPERATURE=0.0
GITHUB_AGENT_MAX_TOKENS=1000
```

### 7.2 Configuration Model

```python
# app/core/config.py (additions)

class GitHubSettings(BaseSettings):
    """GitHub-specific configuration."""
    token: str = Field(..., env="GITHUB_TOKEN")
    default_repo: str = Field(default="", env="GITHUB_DEFAULT_REPO")
    base_url: str = Field(default="https://api.github.com", env="GITHUB_BASE_URL")
    timeout: int = Field(default=30, env="GITHUB_TIMEOUT")
    max_retries: int = Field(default=3, env="GITHUB_MAX_RETRIES")

    class Config:
        env_prefix = "GITHUB_"
```

---

## 8. API Endpoints

### 8.1 New Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/github/query` | Execute GitHub query (sync) |
| POST | `/api/v1/github/chat` | GitHub query via chat (async) |
| GET | `/api/v1/github/repo/{owner}/{repo}/status` | Repository status |
| GET | `/api/v1/github/issues/{owner}/{repo}` | List issues (raw API) |
| POST | `/api/v1/github/issues/{owner}/{repo}` | Create issue (raw API) |

### 8.2 Request/Response Examples

```python
# POST /api/v1/github/chat
{
    "query": "What issues are assigned to @john?",
    "conversation_id": "uuid-123",
    "context": {
        "repo": "acme/backend",
        "branch": "main"
    }
}

# Response 202
{
    "task_id": "celery-task-uuid",
    "status": "processing"
}

# GET /api/v1/messages/{task_id}
{
    "status": "completed",
    "response": {
        "content": "John has 3 issues assigned:\n1. #42: Fix login bug (open)\n2. #45: Update docs (open)\n3. #48: Refactor auth (open)",
        "agent_type": "github",
        "metadata": {
            "intent": "issue:list",
            "issues_count": 3
        }
    }
}
```

---

## 9. Security Considerations

### 9.1 Authentication
- GitHub PAT (Personal Access Token) stored in environment variables
- Never log or expose tokens in error messages
- Support for GitHub App authentication (future enhancement)

### 9.2 Authorization
- Respect GitHub repository permissions
- Return appropriate errors for unauthorized operations
- Validate repository access before operations

### 9.3 Rate Limiting
- Implement client-side rate limit tracking
- Respect GitHub API rate limits (5000/hr authenticated)
- Queue requests when approaching limits

### 9.4 Data Sanitization
- Sanitize error messages before displaying to users
- Remove sensitive data from logged outputs
- Validate all user inputs before GitHub API calls

---

## 10. Monitoring & Observability

### 10.1 Metrics to Track

| Metric | Type | Threshold |
|--------|------|-----------|
| `github_api_requests_total` | Counter | - |
| `github_api_latency_seconds` | Histogram | p95 < 5s |
| `github_api_errors_total` | Counter | < 1% |
| `github_agent_intent_accuracy` | Gauge | > 90% |
| `mcp_tool_calls_total` | Counter | - |
| `mcp_tool_errors_total` | Counter | < 0.5% |

### 10.2 Logging

```python
# Structured logging format
{
    "timestamp": "2026-01-27T10:00:00Z",
    "level": "INFO",
    "service": "github-agent",
    "action": "list_issues",
    "repo": "owner/repo",
    "duration_ms": 1234,
    "status": "success",
    "issues_returned": 5
}
```

---

## 11. Dependencies

### 11.1 New Python Packages

```txt
# requirements.txt additions
aiohttp>=3.9.0              # Async HTTP client for GitHub API
mcp>=0.1.0                  # Model Context Protocol
python-dotenv>=1.0.0        # Environment variable management
```

### 11.2 GitHub API
- GitHub REST API v3
- Minimum required scopes: `repo`, `read:org`

---

## 12. Open Questions

| ID | Question | Impact | Status |
|----|----------|--------|--------|
| OQ-001 | Should we support multiple GitHub repositories per conversation? | Medium | TBD |
| OQ-002 | Do we need webhooks for real-time GitHub event updates? | Low | Out of scope |
| OQ-003 | Should we support GitHub Enterprise Server? | Low | Future |
| OQ-004 | How do we handle very large diffs in PR reviews? | Medium | TBD |
| OQ-005 | Should we implement caching for GitHub API responses? | Medium | TBD |

---

## 13. Appendix

### 13.1 Related Documents
- [Project Constitution](../constitution.md)
- [Agent Guidelines](../AGENTS.md)
- [MCP Specification](https://modelcontextprotocol.io)
- [GitHub REST API Documentation](https://docs.github.com/en/rest)

### 13.2 Glossary

| Term | Definition |
|------|------------|
| **MCP** | Model Context Protocol - Standard for AI tool integration |
| **PAT** | Personal Access Token - GitHub authentication method |
| **LangGraph** | Framework for building stateful AI agents |
| **Celery** | Distributed task queue for background processing |
| **Intent** | The user's underlying goal in a query |

### 13.3 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-27 | Claude Code | Initial spec creation |

---

**Document Status:** 🟢 Draft for Review
**Next Review Date:** 2026-01-30
**Approvals:** Pending
