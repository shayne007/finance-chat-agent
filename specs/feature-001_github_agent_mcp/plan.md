# Implementation Plan: GitHub Chat Agent with MCP Integration

**Feature ID:** feature-001_github_agent_mcp
**Status:** Draft
**Version:** 1.0
**Last Updated:** 2026-01-27

---

## Table of Contents

1. [Overview](#1-overview)
2. [Technical Approach](#2-technical-approach)
3. [Architecture Decisions](#3-architecture-decisions)
4. [Implementation Phases](#4-implementation-phases)
5. [Risk Assessment](#5-risk-assessment)
6. [Resource Requirements](#6-resource-requirements)
7. [Rollout Strategy](#7-rollout-strategy)

---

## 1. Overview

### 1.1 Objective

Implement a production-ready GitHub chat agent using the Model Context Protocol (MCP) that integrates seamlessly with the existing FinanceAgent orchestration system.

### 1.2 Scope

**In Scope:**
- GitHub REST API v3 integration
- MCP server implementation for GitHub tools
- Intent-based query classification
- Issue and Pull Request management
- Repository status and code search
- Integration with existing LangGraph orchestration

**Out of Scope:**
- GitHub Webhooks (future enhancement)
- GraphQL API (REST API sufficient for current requirements)
- GitHub Enterprise Server (cloud-only initially)
- Real-time event streaming

### 1.3 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| GitHub API latency (p95) | < 5 seconds | Application metrics |
| Intent classification accuracy | > 90% | Test suite validation |
| Test coverage | > 95% | Pytest coverage |
| Zero breaking changes | 100% | Existing test suite passes |
| Documentation completeness | 100% | All public APIs documented |

---

## 2. Technical Approach

### 2.1 Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| **HTTP Client** | `aiohttp` | Async I/O, matches FastAPI patterns |
| **MCP Protocol** | `mcp` Python SDK | Official MCP implementation |
| **LLM Integration** | LangChain + OpenAI | Consistent with existing agents |
| **Data Validation** | Pydantic v2 | Already used throughout project |
| **Testing** | pytest + responses | Existing test framework |
| **Configuration** | pydantic-settings | Matches existing config pattern |

### 2.2 Integration Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EXISTING SYSTEM (No Changes)                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    FinanceAgent (LangGraph)                  │  │
│  │   Routes queries to specialized agents based on intent       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                          │                                          │
│                          │ adds new route                           │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     GitHubAgent (NEW)                        │  │
│  │   • Intent classification using LLM                          │  │
│  │   • MCP tool orchestration                                   │  │
│  │   • Response formatting                                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                          │                                          │
│                          │ MCP protocol                             │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   GitHubMCPServer (NEW)                      │  │
│  │   • Implements MCP server protocol                           │  │
│  │   • Exposes GitHub operations as tools                       │  │
│  │   • Handles tool discovery and execution                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                          │                                          │
│                          │ HTTP REST API                            │
│                          ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    GitHubClient (NEW)                        │  │
│  │   • Async GitHub API wrapper                                 │  │
│  │   • Rate limiting and retry logic                            │  │
│  │   • Response parsing and validation                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Design Principles

Following the [Project Constitution](../constitution.md):

1. **Simplicity First**
   - Start with REST API, defer GraphQL
   - Use existing patterns (JiraAgent as reference)
   - No premature optimization

2. **Test-First Development**
   - Write tests before implementation (TDD)
   - Mock GitHub API responses in unit tests
   - Integration tests with test repositories

3. **Type Safety**
   - All public APIs have type hints
   - Pydantic models for all data structures
   - mypy strict mode compliance

4. **Single Responsibility**
   - Separate client, server, and agent concerns
   - Each MCP tool does one thing well
   - Clear interfaces between layers

---

## 3. Architecture Decisions

### 3.1 Decision Record

| ID | Decision | Alternatives | Rationale |
|----|----------|--------------|-----------|
| AD-001 | Use MCP protocol for tool exposure | Direct API calls, custom protocol | Industry standard, enables tool composability |
| AD-002 | Implement separate GitHubClient | Use PyGithub library | More control, async support, lighter dependency |
| AD-003 | LLM-based intent classification | Regex patterns, ML classifier | More flexible, handles natural language better |
| AD-004 | Async/await throughout | Synchronous with thread pool | Matches FastAPI, better resource utilization |
| AD-005 | In-memory MCP server (same process) | Separate MCP server process | Simpler deployment, lower latency |

### 3.2 Component Details

#### 3.2.1 GitHubClient Design

**Responsibilities:**
- Wrap GitHub REST API v3 endpoints
- Handle authentication (Bearer token)
- Implement rate limiting with queue
- Retry logic with exponential backoff
- Response validation using Pydantic models

**Key Methods:**
```python
class GitHubClient:
    async def list_issues(
        self,
        owner: str,
        repo: str,
        state: Literal["open", "closed", "all"] = "open",
        assignee: str | None = None,
        labels: list[str] | None = None,
        milestone: str | None = None,
        per_page: int = 30,
        page: int = 1,
    ) -> list[GitHubIssue]:
        """List repository issues with filters."""

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> GitHubIssue:
        """Create a new issue."""

    async def update_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        title: str | None = None,
        body: str | None = None,
        state: Literal["open", "closed"] | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> GitHubIssue:
        """Update an existing issue."""

    async def close_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        comment: str | None = None,
    ) -> GitHubIssue:
        """Close an issue with optional comment."""

    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: Literal["open", "closed", "all"] = "open",
        head: str | None = None,
        base: str | None = None,
        per_page: int = 30,
        page: int = 1,
    ) -> list[GitHubPullRequest]:
        """List repository pull requests."""

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> GitHubPullRequestDetail:
        """Get detailed pull request information."""

    async def create_pr_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        path: str | None = None,
        position: int | None = None,
    ) -> GitHubComment:
        """Create a comment on a pull request."""

    async def get_repository_status(
        self,
        owner: str,
        repo: str,
        branch: str = "main",
    ) -> RepositoryStatus:
        """Get repository branch status and recent commits."""

    async def get_file_contents(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str | None = None,
    ) -> FileContent:
        """Get file contents from repository."""

    async def search_code(
        self,
        owner: str,
        repo: str,
        query: str,
        per_page: int = 30,
        page: int = 1,
    ) -> list[CodeSearchResult]:
        """Search code within repository."""
```

**Rate Limiting Strategy:**
```python
class GitHubRateLimiter:
    """GitHub API rate limiter with queue management."""

    def __init__(self, requests_per_hour: int = 5000):
        self.requests_per_hour = requests_per_hour
        self.request_times: deque[datetime] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until rate limit allows request."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            # Remove requests older than 1 hour
            cutoff = now - timedelta(hours=1)
            while self.request_times and self.request_times[0] < cutoff:
                self.request_times.popleft()

            if len(self.request_times) >= self.requests_per_hour:
                # Calculate wait time
                oldest_request = self.request_times[0]
                wait_seconds = 3600 - (now - oldest_request).total_seconds()
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
                    # Clean up expired requests after waiting
                    while self.request_times and self.request_times[0] < cutoff:
                        self.request_times.popleft()

            self.request_times.append(now)
```

**Error Handling:**
```python
class GitHubAPIError(Exception):
    """Base exception for GitHub API errors."""

class GitHubRateLimitError(GitHubAPIError):
    """Raised when rate limit is exceeded."""

class GitHubAuthenticationError(GitHubAPIError):
    """Raised when authentication fails."""

class GitHubNotFoundError(GitHubAPIError):
    """Raised when resource is not found."""

class GitHubValidationError(GitHubAPIError):
    """Raised when request validation fails."""
```

#### 3.2.2 MCP Server Design

**Tool Schema Definition:**
```python
from mcp import Tool, ToolContext

class GitHubMCPServer:
    """MCP server exposing GitHub operations as tools."""

    async def list_tools(self) -> list[Tool]:
        """Return list of available GitHub tools."""
        return [
            Tool(
                name="github_list_issues",
                description="List GitHub issues with optional filters",
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"},
                        "state": {
                            "type": "string",
                            "enum": ["open", "closed", "all"],
                            "description": "Issue state filter"
                        },
                        "assignee": {"type": "string", "description": "Filter by assignee"},
                        "labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by labels"
                        },
                    },
                    "required": ["owner", "repo"],
                },
            ),
            Tool(
                name="github_create_issue",
                description="Create a new GitHub issue",
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string"},
                        "repo": {"type": "string"},
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "labels": {"type": "array", "items": {"type": "string"}},
                        "assignees": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["owner", "repo", "title"],
                },
            ),
            # ... additional tools
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        context: ToolContext,
    ) -> dict:
        """Execute a GitHub tool."""
        if name == "github_list_issues":
            return await self._list_issues(arguments)
        elif name == "github_create_issue":
            return await self._create_issue(arguments)
        # ... additional tool handlers
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _list_issues(self, args: dict) -> dict:
        """Handle list_issues tool call."""
        issues = await self.client.list_issues(
            owner=args["owner"],
            repo=args["repo"],
            state=args.get("state", "open"),
            assignee=args.get("assignee"),
            labels=args.get("labels"),
        )
        return {
            "issues": [
                {
                    "id": issue.id,
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "author": issue.author,
                    "assignees": issue.assignees,
                    "labels": issue.labels,
                    "url": issue.url,
                }
                for issue in issues
            ]
        }
```

#### 3.2.3 GitHubAgent Design

**Intent Classification:**
```python
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

class GitHubAgent:
    """Agent for handling GitHub operations."""

    INTENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a GitHub query classifier. Classify the user's intent into one of these categories:

- issue:list: User wants to list/query issues
- issue:create: User wants to create a new issue
- issue:update: User wants to update an existing issue
- issue:close: User wants to close an issue
- pr:list: User wants to list/query pull requests
- pr:view: User wants to view PR details
- pr:comment: User wants to comment on a PR
- repo:status: User wants repository/branch status
- file:get: User wants to retrieve file contents
- code:search: User wants to search code
- general: General GitHub question

Extract entities like:
- repo: repository name (owner/repo format)
- assignee: GitHub username
- labels: list of labels
- state: open/closed/all
- branch: branch name
- pr_number: pull request number
- issue_number: issue number
- path: file path

{format_instructions}"""),
        ("human", "{query}")
    ])

    def __init__(
        self,
        llm: ChatOpenAI,
        mcp_server: GitHubMCPServer,
        default_repo: str | None = None,
    ):
        self.llm = llm
        self.mcp_server = mcp_server
        self.default_repo = default_repo
        self.parser = PydanticOutputParser(pydantic_object=GitHubIntent)

    async def classify_intent(self, query: str) -> GitHubIntent:
        """Classify the intent of a GitHub query."""
        prompt = self.INTENT_PROMPT.format(
            query=query,
            format_instructions=self.parser.get_format_instructions(),
        )
        response = await self.llm.ainvoke(prompt)
        return self.parser.parse(response)

    async def process_query(
        self,
        query: str,
        context: dict,
    ) -> AgentResponse:
        """Process a GitHub query and return response."""
        try:
            # Step 1: Classify intent
            intent = await self.classify_intent(query)

            # Step 2: Extract context
            repo = context.get("repo") or intent.entities.get("repo") or self.default_repo
            if not repo:
                return AgentResponse(
                    success=False,
                    content="I need to know which repository to query. Please specify the repository (e.g., 'owner/repo').",
                    error="Missing repository",
                )

            # Step 3: Route to appropriate MCP tool
            tool_result = await self._route_to_tool(intent, repo, context)

            # Step 4: Format response
            response = self._format_response(tool_result, intent)

            return AgentResponse(
                success=True,
                content=response,
                metadata={"intent": intent, "repo": repo},
            )

        except Exception as e:
            logger.exception("Error processing GitHub query")
            return AgentResponse(
                success=False,
                content=f"Sorry, I encountered an error: {str(e)}",
                error=str(e),
            )

    async def _route_to_tool(
        self,
        intent: GitHubIntent,
        repo: str,
        context: dict,
    ) -> dict:
        """Route intent to appropriate MCP tool."""
        owner, repo_name = repo.split("/", 1)

        if intent.category == "issue" and intent.action == "list":
            return await self.mcp_server.call_tool(
                "github_list_issues",
                {
                    "owner": owner,
                    "repo": repo_name,
                    "state": intent.entities.get("state", "open"),
                    "assignee": intent.entities.get("assignee"),
                    "labels": intent.entities.get("labels"),
                },
                context,
            )

        elif intent.category == "issue" and intent.action == "create":
            return await self.mcp_server.call_tool(
                "github_create_issue",
                {
                    "owner": owner,
                    "repo": repo_name,
                    "title": intent.entities.get("title"),
                    "body": intent.entities.get("body"),
                    "labels": intent.entities.get("labels"),
                    "assignees": intent.entities.get("assignees"),
                },
                context,
            )

        # ... additional intent handlers

        else:
            raise ValueError(f"Unsupported intent: {intent.category}:{intent.action}")

    def _format_response(self, tool_result: dict, intent: GitHubIntent) -> str:
        """Format tool result into conversational response."""
        if intent.category == "issue" and intent.action == "list":
            issues = tool_result.get("issues", [])
            if not issues:
                return "No issues found."

            formatted = []
            for issue in issues[:10]:  # Limit to 10 issues
                formatted.append(
                    f"#{issue['number']}: {issue['title']} "
                    f"({issue['state']}) - {issue['author']}"
                )
            return f"Found {len(issues)} issue(s):\n" + "\n".join(formatted)

        # ... additional formatters

        else:
            return str(tool_result)
```

### 3.3 FinanceAgent Integration

**Route Addition:**
```python
# app/agents/finance_agent.py

class FinanceAgent:
    """Main orchestrator agent."""

    def __init__(
        self,
        jira_agent: JiraAgent,
        rag_agent: RAGAgent,
        github_agent: GitHubAgent,  # NEW
    ):
        self.jira_agent = jira_agent
        self.rag_agent = rag_agent
        self.github_agent = github_agent  # NEW

    async def route_agent(self, query: str, context: dict) -> AgentResponse:
        """Route query to appropriate agent."""
        # Check for GitHub keywords
        github_keywords = [
            "issue", "pr", "pull request", "github",
            "commit", "branch", "repository", "repo",
        ]
        if any(keyword in query.lower() for keyword in github_keywords):
            logger.info("Routing to GitHubAgent")
            return await self.github_agent.process_query(query, context)

        # Existing Jira routing
        jira_keywords = ["ticket", "jira"]
        if any(keyword in query.lower() for keyword in jira_keywords):
            logger.info("Routing to JiraAgent")
            return await self.jira_agent.process_query(query, context)

        # Default to RAG or general LLM
        # ... existing logic
```

---

## 4. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goal:** Build GitHub client with core issue operations

#### Tasks

| ID | Task | File | Description | Acceptance Criteria |
|----|------|------|-------------|---------------------|
| P1-T1 | Create GitHub models | `app/models/github.py` | Define Pydantic models | All models validate correctly |
| P1-T2 | Create GitHub client | `app/clients/github_client.py` | Implement client skeleton | Client initializes with token |
| P1-T3 | Implement list_issues | `app/clients/github_client.py` | Add issues list method | Returns list of GitHubIssue |
| P1-T4 | Implement create_issue | `app/clients/github_client.py` | Add issue creation | Creates issue and returns GitHubIssue |
| P1-T5 | Add rate limiting | `app/clients/github_client.py` | Implement rate limiter | Respects GitHub rate limits |
| P1-T6 | Add error handling | `app/clients/github_client.py` | Custom exceptions | All error paths handled |
| P1-T7 | Write unit tests | `tests/clients/test_github_client.py` | Test client methods | 90%+ coverage |
| P1-T8 | Add config | `app/core/config.py` | GitHub settings | Configurable via env vars |

**Files to Create:**
```
app/models/github.py
app/clients/github_client.py
tests/clients/test_github_client.py
```

**Files to Modify:**
```
app/core/config.py
requirements.txt
pyproject.toml (if using)
```

**Dependencies to Add:**
```txt
aiohttp>=3.9.0
```

---

### Phase 2: MCP Server (Week 2-3)

**Goal:** Implement MCP server exposing GitHub tools

#### Tasks

| ID | Task | File | Description | Acceptance Criteria |
|----|------|------|-------------|---------------------|
| P2-T1 | Create MCP server | `app/mcp/github_server.py` | Server skeleton | Server starts without errors |
| P2-T2 | Implement list_tools | `app/mcp/github_server.py` | Tool discovery | Returns all GitHub tools |
| P2-T3 | Implement call_tool | `app/mcp/github_server.py` | Tool execution | Executes tools correctly |
| P2-T4 | Add issue tools | `app/mcp/github_server.py` | Issue MCP tools | list_issues, create_issue work |
| P2-T5 | Add error handling | `app/mcp/github_server.py` | Tool error handling | Graceful error responses |
| P2-T6 | Write tests | `tests/mcp/test_github_server.py` | Test MCP server | 85%+ coverage |
| P2-T7 | Add config | `app/core/config.py` | MCP settings | Server configurable |

**Files to Create:**
```
app/mcp/__init__.py
app/mcp/github_server.py
tests/mcp/test_github_server.py
```

**Dependencies to Add:**
```txt
mcp>=0.1.0
```

---

### Phase 3: GitHub Agent (Week 3-4)

**Goal:** Build conversational agent with intent classification

#### Tasks

| ID | Task | File | Description | Acceptance Criteria |
|----|------|------|-------------|---------------------|
| P3-T1 | Create agent | `app/agents/github_agent.py` | Agent skeleton | Agent initializes |
| P3-T2 | Implement intent classifier | `app/agents/github_agent.py` | LLM-based classification | >90% accuracy on test set |
| P3-T3 | Add query routing | `app/agents/github_agent.py` | Route to MCP tools | Correct tool called |
| P3-T4 | Format responses | `app/agents/github_agent.py` | Conversational output | Natural language responses |
| P3-T5 | Add error handling | `app/agents/github_agent.py` | Graceful failures | User-friendly errors |
| P3-T6 | Write tests | `tests/agents/test_github_agent.py` | Test agent logic | 90%+ coverage |

**Files to Create:**
```
app/agents/github_agent.py
tests/agents/test_github_agent.py
tests/fixtures/github_intent_fixtures.py
```

---

### Phase 4: FinanceAgent Integration (Week 4-5)

**Goal:** Integrate GitHubAgent into existing orchestration

#### Tasks

| ID | Task | File | Description | Acceptance Criteria |
|----|------|------|-------------|---------------------|
| P4-T1 | Update route logic | `app/agents/finance_agent.py` | Add GitHub routing | GitHub queries route correctly |
| P4-T2 | Update DI container | `app/core/container.py` | Register GitHubAgent | Agent injectable |
| P4-T3 | Add API endpoints | `app/api/v1/github.py` | GitHub-specific routes | Endpoints functional |
| P4-T4 | Update Celery tasks | `app/tasks/message_tasks.py` | Handle GitHub queries | Async processing works |
| P4-T5 | Integration tests | `tests/integration/test_github_flow.py` | E2E tests | Full flow tested |

**Files to Create:**
```
app/api/v1/github.py
tests/integration/test_github_flow.py
```

**Files to Modify:**
```
app/agents/finance_agent.py
app/core/container.py (or similar DI setup)
app/tasks/message_tasks.py
app/api/v1/router.py
```

---

### Phase 5: Advanced Features (Week 5-6)

**Goal:** Add PR operations, search, and repo status

#### Tasks

| ID | Task | File | Description | Acceptance Criteria |
|----|------|------|-------------|---------------------|
| P5-T1 | PR operations | `app/clients/github_client.py` | Add PR methods | List, view PRs work |
| P5-T2 | PR MCP tools | `app/mcp/github_server.py` | Expose PR tools | PR tools functional |
| P5-T3 | Code search | `app/clients/github_client.py` | Add search method | Search returns results |
| P5-T4 | Search MCP tool | `app/mcp/github_server.py` | Expose search | Search tool works |
| P5-T5 | Repo status | `app/clients/github_client.py` | Add status method | Returns branch info |
| P5-T6 | Status MCP tool | `app/mcp/github_server.py` | Expose status | Status tool works |
| P5-T7 | Update agent | `app/agents/github_agent.py` | Handle new intents | All intents work |
| P5-T8 | Tests | `tests/` | Test all features | All tests pass |

---

### Phase 6: Documentation & Hardening (Week 6-7)

**Goal:** Production readiness

#### Tasks

| ID | Task | Description | Acceptance Criteria |
|----|------|-------------|---------------------|
| P6-T1 | API documentation | Document all public APIs | All APIs documented |
| P6-T2 | User guide | Write command examples | Users can query GitHub |
| P6-T3 | Performance testing | Run load tests | p95 < 5s achieved |
| P6-T4 | Security audit | Review token handling | No token exposure |
| P6-T5 | Error refinement | Improve error messages | Clear user feedback |
| P6-T6 | Monitoring setup | Add metrics/metrics | All metrics emitted |
| P6-T7 | Deployment guide | Document deployment | Deployable to prod |

---

## 5. Risk Assessment

### 5.1 Risk Register

| ID | Risk | Probability | Impact | Mitigation |
|----|------|-------------|--------|------------|
| R-001 | GitHub API rate limits | Medium | Medium | Implement queue + tracking, provide user feedback |
| R-002 | Intent classification accuracy | Medium | High | Comprehensive test set, fallback to general LLM |
| R-003 | Breaking existing agents | Low | High | Comprehensive regression tests, feature flags |
| R-004 | MCP SDK maturity | Low | Medium | Evaluate SDK stability, have fallback plan |
| R-005 | Token security breach | Low | Critical | Secure storage, audit logs, rotation policy |
| R-006 | Performance degradation | Low | Medium | Load testing, caching, optimization |
| R-007 | Integration complexity | Medium | Medium | Incremental integration, thorough testing |

### 5.2 Risk Mitigation Plans

#### R-001: GitHub API Rate Limits
**Mitigation:**
```python
class RateLimitTracker:
    """Track and manage GitHub API rate limits."""

    def __init__(self, limit: int = 5000):
        self.limit = limit
        self.used = 0
        self.reset_at: datetime | None = None

    async def check_limit(self) -> bool:
        """Check if we can make a request."""
        if self.reset_at and datetime.now(timezone.utc) >= self.reset_at:
            self.used = 0
            self.reset_at = None
        return self.used < self.limit

    async def wait_if_needed(self) -> None:
        """Wait if rate limit is approached."""
        while not await self.check_limit():
            wait_seconds = (self.reset_at - datetime.now(timezone.utc)).total_seconds()
            logger.warning(f"Rate limit reached, waiting {wait_seconds}s")
            await asyncio.sleep(min(wait_seconds, 60))
```

#### R-002: Intent Classification Accuracy
**Mitigation:**
- Build comprehensive test dataset (100+ example queries)
- Use few-shot prompting with examples
- Implement confidence threshold with fallback
- Allow manual correction for learning

#### R-003: Breaking Existing Agents
**Mitigation:**
- Feature flag for GitHub agent
- Comprehensive regression test suite
- Gradual rollout (canary deployment)
- Monitoring for error rate increases

---

## 6. Resource Requirements

### 6.1 Development Resources

| Role | Allocation | Responsibilities |
|------|------------|-------------------|
| **Backend Developer** | 1.0 FTE (6 weeks) | Implementation of all phases |
| **QA Engineer** | 0.5 FTE (4 weeks) | Test case development, execution |
| **DevOps Engineer** | 0.2 FTE (2 weeks) | Deployment, monitoring setup |

### 6.2 Infrastructure

| Resource | Specification | Purpose |
|----------|---------------|---------|
| **GitHub Account** | Test repository with sample data | Integration testing |
| **GitHub PAT** | Classic token with `repo` scope | API authentication |
| **Database** | Existing PostgreSQL | Store GitHub metadata if needed |
| **Redis** | Existing instance | Celery broker, caching |
| **Monitoring** | Existing Prometheus/Grafana | Metrics dashboards |

### 6.3 External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| aiohttp | >=3.9.0 | Async HTTP client |
| mcp | >=0.1.0 | MCP protocol implementation |
| langchain-openai | existing | LLM integration |
| pytest | existing | Testing |
| responses | existing | Mock HTTP responses |

---

## 7. Rollout Strategy

### 7.1 Deployment Phases

#### Stage 1: Development (Week 1-6)
- Feature branch: `feature/github-agent-mcp`
- Development environment testing
- Internal team validation

#### Stage 2: Staging (Week 6-7)
- Deploy to staging environment
- Load testing with production-like data
- Security audit
- Performance validation

#### Stage 3: Canary (Week 7)
- Deploy to production with 10% traffic
- Monitor error rates, latency
- Rollback plan ready

#### Stage 4: Full Rollout (Week 7-8)
- Gradually increase to 100% traffic
- Continuous monitoring
- User feedback collection

### 7.2 Rollback Plan

**Conditions for Rollback:**
- Error rate > 1% for 5 minutes
- P95 latency > 10 seconds for 5 minutes
- Any critical security issue identified
- Breaking changes to existing agents detected

**Rollback Steps:**
1. Disable GitHub agent via feature flag
2. Verify existing agents恢复正常
3. Investigate and fix issues
4. Re-deploy to staging for validation

### 7.3 Feature Flags

```python
# app/core/config.py

class GitHubSettings(BaseSettings):
    """GitHub agent configuration."""

    # Feature flag
    enabled: bool = Field(default=False, env="GITHUB_AGENT_ENABLED")

    # Canary deployment
    traffic_percentage: int = Field(default=0, env="GITHUB_AGENT_TRAFFIC_PCT")

    # Whitelist users (for gradual rollout)
    whitelist_users: list[str] = Field(default_factory=list, env="GITHUB_AGENT_WHITELIST_USERS")
```

### 7.4 Monitoring

**Key Metrics to Monitor:**
```python
# Metrics to track
- github_agent_requests_total
- github_agent_errors_total
- github_agent_latency_seconds (histogram)
- github_api_rate_limit_remaining (gauge)
- github_mcp_tool_calls_total (counter by tool name)
- github_intent_classification_accuracy (gauge)
```

**Alerts:**
```yaml
alerts:
  - name: HighGitHubAgentErrorRate
    condition: error_rate > 0.01
    duration: 5m
    severity: warning

  - name: GitHubAgentHighLatency
    condition: p95_latency > 10s
    duration: 5m
    severity: warning

  - name: GitHubRateLimitNear
    condition: rate_limit_remaining < 100
    duration: 1m
    severity: info
```

---

## 8. Open Questions Tracking

| ID | Question | Raised By | Target Resolution | Status |
|----|----------|-----------|-------------------|--------|
| OQ-001 | Multi-repository support? | - | Phase 5 | Open |
| OQ-002 | Large diff handling? | - | Phase 5 | Open |
| OQ-003 | Caching strategy? | - | Phase 6 | Open |

---

## 9. Appendix

### 9.1 Code Organization

```
finance-chat-agent/
├── app/
│   ├── agents/
│   │   ├── finance_agent.py         # MODIFY: Add GitHub routing
│   │   ├── jira_agent.py            # EXISTING
│   │   ├── rag_agent.py             # EXISTING
│   │   └── github_agent.py          # NEW: GitHub agent
│   ├── api/
│   │   └── v1/
│   │       ├── router.py            # MODIFY: Add GitHub routes
│   │       └── github.py            # NEW: GitHub endpoints
│   ├── clients/
│   │   ├── jira_client.py           # EXISTING
│   │   └── github_client.py         # NEW: GitHub API client
│   ├── core/
│   │   ├── config.py                # MODIFY: Add GitHub config
│   │   └── container.py             # MODIFY: Add GitHubAgent DI
│   ├── mcp/                         # NEW: MCP servers
│   │   ├── __init__.py
│   │   └── github_server.py         # NEW: GitHub MCP server
│   ├── models/
│   │   └── github.py                # NEW: GitHub data models
│   └── tasks/
│       └── message_tasks.py         # MODIFY: Handle GitHub queries
├── tests/
│   ├── agents/
│   │   └── test_github_agent.py     # NEW
│   ├── clients/
│   │   └── test_github_client.py    # NEW
│   ├── mcp/
│   │   └── test_github_server.py    # NEW
│   ├── integration/
│   │   └── test_github_flow.py      # NEW
│   └── fixtures/
│       └── github_intent_fixtures.py # NEW
└── specs/
    └── feature-001_github_agent_mcp/
        ├── spec.md                   # COMPLETE
        ├── plan.md                   # THIS FILE
        └── tasks.md                  # TODO: Actionable tasks
```

### 9.2 Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/github-agent-mcp

# 2. Set up environment
cp .env.example .env
# Add GITHUB_TOKEN to .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
pytest tests/ -v

# 5. Run development server
uvicorn app.main:app --reload

# 6. Test endpoints
curl -X POST http://localhost:8000/api/v1/github/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What issues are open?"}'

# 7. Commit and push
git add .
git commit -m "feat: implement GitHub agent foundation"
git push origin feature/github-agent-mcp
```

### 9.3 Testing Commands

```bash
# Unit tests
pytest tests/clients/test_github_client.py -v
pytest tests/mcp/test_github_server.py -v
pytest tests/agents/test_github_agent.py -v

# Integration tests
pytest tests/integration/test_github_flow.py -v

# With coverage
pytest --cov=app/clients/github_client --cov=app/mcp/github_server --cov=app/agents/github_agent

# Specific test
pytest tests/agents/test_github_agent.py::test_intent_classification -v

# Watch mode
pytest-watch tests/
```

---

**Document Status:** 🟡 Draft for Technical Review
**Next Actions:**
1. Review and approve technical approach
2. Create detailed tasks.md with actionable items
3. Set up development environment
4. Begin Phase 1 implementation

---

**Version History**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-27 | Claude Code | Initial implementation plan |
