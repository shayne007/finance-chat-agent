# Tasks: GitHub Chat Agent with MCP Integration

**Feature ID:** feature-001_github_agent_mcp
**Status:** Not Started
**Version:** 2.0
**Last Updated:** 2026-01-27

---

## TDD Development Process (Per Constitution)

> **Constitution Article 2: Test-First Imperative - Non-Negotiable**
>
> All new features or bug fixes must begin with writing one (or more) failing tests.
>
> Follow the **"Red-Green-Refactor"** cycle:
> 1. 🔴 **RED**: Write failing test(s) first
> 2. 🟢 **GREEN**: Make test(s) pass with minimal implementation
> 3. 🔵 **REFACTOR**: Improve code while keeping tests green
>
> - Use `@pytest.mark.parametrize` for multiple inputs/edge cases
> - Prioritize integration tests over excessive mocking
> - Never skip the "write failing test first" step

### TDD Task Legend

| Symbol | Phase | Description |
|--------|-------|-------------|
| 🔴 | RED | Write failing test(s) |
| 🟢 | GREEN | Implement to make tests pass |
| 🔵 | REFACTOR | Refactor while tests pass |
| ⚙️ | SETUP | Environment/setup tasks (TDD not applicable) |

---

## Task Overview

- **Total Tasks:** 258
- **Completed:** 0
- **In Progress:** 0
- **Pending:** 258

---

## Phase 1: Foundation (Week 1-2)

### 1.0 Environment Setup (TDD N/A)

- [ ] ⚙️ **P1-S1** Add `aiohttp>=3.9.0` to requirements.txt
- [ ] ⚙️ **P1-S2** Add `mcp>=0.1.0` to requirements.txt
- [ ] ⚙️ **P1-S3** Update requirements.txt with version pinning
- [ ] ⚙️ **P1-S4** Run `pip install -r requirements.txt` in dev environment
- [ ] ⚙️ **P1-S5** Verify all dependencies install without conflicts
- [ ] ⚙️ **P1-S6** Create requirements-dev.txt if needed for testing dependencies

---

### 1.1 GitHub Models (TDD Cycle)

#### Feature: Pydantic Models for GitHub Data

- [ ] 🔴 **RED** Create test file `tests/models/test_github.py`
- [ ] 🔴 **RED** Write parameterized test for `GitHubIssue` validation with valid data
- [ ] 🔴 **RED** Write parameterized test for `GitHubIssue` validation with invalid data (missing required fields)
- [ ] 🔴 **RED** Write parameterized test for `GitHubIssue` validation with invalid data (wrong types)
- [ ] 🔴 **RED** Write parameterized test for `GitHubPullRequest` validation
- [ ] 🔴 **RED** Write parameterized test for `GitHubComment` validation
- [ ] 🔴 **RED** Write parameterized test for `RepositoryStatus` validation
- [ ] 🔴 **RED** Write parameterized test for `FileContent` validation
- [ ] 🔴 **RED** Write parameterized test for `CodeSearchResult` validation
- [ ] 🔴 **RED** Write parameterized test for `GitHubIntent` validation
- [ ] 🔴 **RED** Write parameterized test for `AgentResponse` validation
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Create `app/models/github.py` file
- [ ] 🟢 **GREEN** Define `GitHubIssue` Pydantic model (id, number, title, body, state, author, assignees, labels, created_at, updated_at, url)
- [ ] 🟢 **GREEN** Define `GitHubPullRequest` Pydantic model (id, number, title, body, state, author, head_branch, base_branch, created_at, url)
- [ ] 🟢 **GREEN** Define `GitHubComment` Pydantic model (id, body, author, created_at, url)
- [ ] 🟢 **GREEN** Define `RepositoryStatus` Pydantic model (branch, commit sha, status checks)
- [ ] 🟢 **GREEN** Define `FileContent` Pydantic model (path, content, sha, size)
- [ ] 🟢 **GREEN** Define `CodeSearchResult` Pydantic model (path, score, matches)
- [ ] 🟢 **GREEN** Define `GitHubIntent` Pydantic model (category, action, confidence, entities)
- [ ] 🟢 **GREEN** Define `AgentResponse` Pydantic model (success, content, metadata, error)
- [ ] 🟢 **GREEN** Run tests, verify all pass
- [ ] 🔵 **REFACTOR** Review models for any simplification opportunities
- [ ] 🔵 **REFACTOR** Add computed properties if useful (e.g., `is_open` on GitHubIssue)
- [ ] 🔵 **REFACTOR** Run tests after refactoring, verify still pass
- [ ] 🔵 **REFACTOR** Run mypy strict type checking on models

---

### 1.2 GitHub Exceptions (TDD Cycle)

#### Feature: Custom Exception Hierarchy

- [ ] 🔴 **RED** Create test file `tests/clients/test_github_exceptions.py`
- [ ] 🔴 **RED** Write test for `GitHubAPIError` can be raised and caught
- [ ] 🔴 **RED** Write test for `GitHubRateLimitError` inherits from `GitHubAPIError`
- [ ] 🔴 **RED** Write test for `GitHubAuthenticationError` inherits from `GitHubAPIError`
- [ ] 🔴 **RED** Write test for `GitHubNotFoundError` inherits from `GitHubAPIError`
- [ ] 🔴 **RED** Write test for `GitHubValidationError` inherits from `GitHubAPIError`
- [ ] 🔴 **RED** Write parameterized test that all exceptions can be caught as `GitHubAPIError`
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Create exception classes in `app/clients/github_client.py`
- [ ] 🟢 **GREEN** Implement `GitHubAPIError` base exception class
- [ ] 🟢 **GREEN** Implement `GitHubRateLimitError` exception class
- [ ] 🟢 **GREEN** Implement `GitHubAuthenticationError` exception class
- [ ] 🟢 **GREEN** Implement `GitHubNotFoundError` exception class
- [ ] 🟢 **GREEN** Implement `GitHubValidationError` exception class
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 1.3 GitHub Rate Limiter (TDD Cycle)

#### Feature: Rate Limiting with Sliding Window

- [ ] 🔴 **RED** Write parameterized test for rate limiter below limit (should allow immediately)
- [ ] 🔴 **RED** Write parameterized test for rate limiter at exact limit (should allow)
- [ ] 🔴 **RED** Write parameterized test for rate limiter above limit (should wait)
- [ ] 🔴 **RED** Write test for sliding window cleanup (old requests expire)
- [ ] 🔴 **RED** Write test for concurrent acquire requests
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Create `GitHubRateLimiter` class in `app/clients/github_client.py`
- [ ] 🟢 **GREEN** Implement `__init__` method with requests_per_hour parameter
- [ ] 🟢 **GREEN** Implement `acquire` async method with rate limit tracking
- [ ] 🟢 **GREEN** Add time window sliding logic (1 hour window)
- [ ] 🟢 **GREEN** Add wait logic when limit is reached
- [ ] 🟢 **GREEN** Add cleanup of expired requests after waiting
- [ ] 🟢 **GREEN** Run tests, verify all pass
- [ ] 🔵 **REFACTOR** Review if logic can be simplified
- [ ] 🔵 **REFACTOR** Add logging for when rate limit wait occurs
- [ ] 🔵 **REFACTOR** Run tests after refactoring

---

### 1.4 GitHub Client - Core (TDD Cycle)

#### Feature: HTTP Client with Authentication

- [ ] 🔴 **RED** Write test for client initialization with token
- [ ] 🔴 **RED** Write test for client initialization with custom base_url
- [ ] 🔴 **RED** Write parameterized test for `_make_request` adds authentication header
- [ ] 🔴 **RED** Write parameterized test for `_make_request` handles 200 OK response
- [ ] 🔴 **RED** Write parameterized test for `_make_request` handles 404 Not Found (raises `GitHubNotFoundError`)
- [ ] 🔴 **RED** Write parameterized test for `_make_request` handles 401 Unauthorized (raises `GitHubAuthenticationError`)
- [ ] 🔴 **RED** Write parameterized test for `_make_request` handles 429 Rate Limit (raises `GitHubRateLimitError`)
- [ ] 🔴 **RED** Write parameterized test for `_make_request` handles 5xx errors (raises `GitHubAPIError`)
- [ ] 🔴 **RED** Write test for `_make_request` handles invalid JSON response
- [ ] 🔴 **RED** Write test for `_make_request` integrates rate limiter
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Create `app/clients/github_client.py` file
- [ ] 🟢 **GREEN** Create `GitHubClient` class with `__init__` method (token, base_url, timeout, max_retries)
- [ ] 🟢 **GREEN** Implement private `_make_request` method with aiohttp
- [ ] 🟢 **GREEN** Add authentication header setup in `_make_request`
- [ ] 🟢 **GREEN** Add response status code handling in `_make_request`
- [ ] 🟢 **GREEN** Add JSON parsing with error handling in `_make_request`
- [ ] 🟢 **GREEN** Integrate rate limiter into `GitHubClient._make_request`
- [ ] 🟢 **GREEN** Run tests, verify all pass
- [ ] 🔵 **REFACTOR** Extract error mapping to separate method if complex
- [ ] 🔵 **REFACTOR** Run tests after refactoring

---

### 1.5 GitHub Client - Issue Operations (TDD Cycle)

#### Feature: List Issues

- [ ] 🔴 **RED** Write parameterized test for `list_issues` with various filters (state, assignee, labels)
- [ ] 🔴 **RED** Write test for `list_issues` with pagination (page, per_page)
- [ ] 🔴 **RED** Write test for `list_issues` returns empty list when no issues
- [ ] 🔴 **RED** Write test for `list_issues` parses response into `GitHubIssue` models
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `list_issues` method signature with type hints
- [ ] 🟢 **GREEN** Add query parameters construction (state, assignee, labels, milestone)
- [ ] 🟢 **GREEN** Add pagination support (per_page, page parameters)
- [ ] 🟢 **GREEN** Parse response into list of `GitHubIssue` models
- [ ] 🟢 **GREEN** Handle empty results gracefully
- [ ] 🟢 **GREEN** Run tests, verify all pass
- [ ] 🔵 **REFACTOR** Run tests after refactoring

#### Feature: Create Issue

- [ ] 🔴 **RED** Write parameterized test for `create_issue` with minimal required fields
- [ ] 🔴 **RED** Write parameterized test for `create_issue` with all optional fields
- [ ] 🔴 **RED** Write test for `create_issue` returns `GitHubIssue` model
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `create_issue` method signature
- [ ] 🟢 **GREEN** Add request body construction (title, body, labels, assignees)
- [ ] 🟢 **GREEN** Parse response into `GitHubIssue` model
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: Update Issue

- [ ] 🔴 **RED** Write parameterized test for `update_issue` with various update fields
- [ ] 🔴 **RED** Write test for `update_issue` with partial updates
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `update_issue` method signature
- [ ] 🟢 **GREEN** Add PATCH request logic for issue updates
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: Close Issue

- [ ] 🔴 **RED** Write test for `close_issue` without comment
- [ ] 🔴 **RED** Write test for `close_issue` with comment
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `close_issue` method
- [ ] 🟢 **GREEN** Add optional comment when closing issue
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Integration Test

- [ ] 🔴 **RED** Write integration test with real GitHub test repository for `list_issues`
- [ ] 🔴 **RED** Write integration test with real GitHub test repository for `create_issue`
- [ ] 🔴 **RED** Set up GitHub test repository and token for integration tests
- [ ] 🟢 **GREEN** Run integration tests with real GitHub API
- [ ] 🔵 **REFACTOR** Clean up test data from integration tests

---

### 1.6 Configuration (TDD Cycle)

#### Feature: GitHub Settings

- [ ] 🔴 **RED** Write test for `GitHubSettings` loads from environment variables
- [ ] 🔴 **RED** Write parameterized test for missing required fields (raises ValidationError)
- [ ] 🔴 **RED** Write test for token format validation (starts with `ghp_` or similar pattern)
- [ ] 🔴 **RED** Write test for default values when optional fields not provided
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `GitHubSettings` class to `app/core/config.py`
- [ ] 🟢 **GREEN** Add `token` field with env var `GITHUB_TOKEN`
- [ ] 🟢 **GREEN** Add `default_repo` field with env var `GITHUB_DEFAULT_REPO`
- [ ] 🟢 **GREEN** Add `base_url` field with env var `GITHUB_BASE_URL`
- [ ] 🟢 **GREEN** Add `timeout` field with env var `GITHUB_TIMEOUT`
- [ ] 🟢 **GREEN** Add `max_retries` field with env var `GITHUB_MAX_RETRIES`
- [ ] 🟢 **GREEN** Add validation for token format
- [ ] 🟢 **GREEN** Add `enabled` field with env var `GITHUB_AGENT_ENABLED`
- [ ] 🟢 **GREEN** Run tests, verify all pass
- [ ] ⚙️ **SETUP** Update `.env.example` with GitHub configuration

---

## Phase 2: MCP Server (Week 2-3)

### 2.1 MCP Server Foundation (TDD Cycle)

#### Feature: MCP Server Initialization

- [ ] 🔴 **RED** Create test directory `tests/mcp/` and `__init__.py`
- [ ] 🔴 **RED** Write test for `GitHubMCPServer` initializes with GitHubClient dependency
- [ ] 🔴 **RED** Write test for `GitHubMCPServer` starts without errors
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] ⚙️ **SETUP** Create `app/mcp/` directory
- [ ] ⚙️ **SETUP** Create `app/mcp/__init__.py` file
- [ ] ⚙️ **SETUP** Create `app/mcp/github_server.py` file
- [ ] ⚙️ **SETUP** Import MCP SDK components
- [ ] 🟢 **GREEN** Create `GitHubMCPServer` class skeleton
- [ ] 🟢 **GREEN** Implement `__init__` method with GitHubClient dependency
- [ ] 🟢 **GREEN** Add MCP server initialization in `__init__`
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 2.2 MCP Tool Discovery (TDD Cycle)

#### Feature: List Tools

- [ ] 🔴 **RED** Write parameterized test for `list_tools` returns all GitHub tools
- [ ] 🔴 **RED** Write test for `github_list_issues` tool schema (name, description, input_schema)
- [ ] 🔴 **RED** Write test for `github_create_issue` tool schema
- [ ] 🔴 **RED** Write test for `github_update_issue` tool schema
- [ ] 🔴 **RED** Write test for `github_close_issue` tool schema
- [ ] 🔴 **RED** Write parameterized test for tool schemas have required fields
- [ ] 🔴 **RED** Write parameterized test for tool schemas have correct field types
- [ ] 🔴 **RED** Write test for tool schemas are valid JSON Schema
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `list_tools` async method
- [ ] 🟢 **GREEN** Define `github_list_issues` tool schema
- [ ] 🟢 **GREEN** Define `github_create_issue` tool schema
- [ ] 🟢 **GREEN** Define `github_update_issue` tool schema
- [ ] 🟢 **GREEN** Define `github_close_issue` tool schema
- [ ] 🟢 **GREEN** Add all required fields to input schemas
- [ ] 🟢 **GREEN** Add optional fields to input schemas with proper types
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 2.3 MCP Tool Execution (TDD Cycle)

#### Feature: Call Tool - List Issues

- [ ] 🔴 **RED** Write parameterized test for `call_tool` with `github_list_issues` and valid arguments
- [ ] 🔴 **RED** Write test for `call_tool` with `github_list_issues` returns issue list
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `call_tool` async method signature
- [ ] 🟢 **GREEN** Add tool name routing logic
- [ ] 🟢 **GREEN** Implement `_list_issues` handler method
- [ ] 🟢 **GREEN** Call `GitHubClient.list_issues` from handler
- [ ] 🟢 **GREEN** Transform GitHubIssue models to dict format
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: Call Tool - Create Issue

- [ ] 🔴 **RED** Write parameterized test for `call_tool` with `github_create_issue`
- [ ] 🔴 **RED** Write test for `call_tool` with `github_create_issue` returns created issue
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `_create_issue` handler method
- [ ] 🟢 **GREEN** Call `GitHubClient.create_issue` from handler
- [ ] 🟢 **GREEN** Transform result to dict format
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: Call Tool - Update/Close Issue

- [ ] 🔴 **RED** Write parameterized test for `call_tool` with `github_update_issue`
- [ ] 🔴 **RED** Write parameterized test for `call_tool` with `github_close_issue`
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `_update_issue` handler method
- [ ] 🟢 **GREEN** Implement `_close_issue` handler method
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: Error Handling

- [ ] 🔴 **RED** Write test for `call_tool` with unknown tool name raises `ValueError`
- [ ] 🔴 **RED** Write test for `call_tool` with invalid arguments raises error
- [ ] 🔴 **RED** Write parameterized test for GitHub exceptions transform to MCP errors
- [ ] 🔴 **RED** Write test for error messages are user-friendly
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `ValueError` for unknown tool names
- [ ] 🟢 **GREEN** Add try-except blocks in tool handlers
- [ ] 🟢 **GREEN** Catch `GitHubAPIError` and transform to MCP error format
- [ ] 🟢 **GREEN** Catch `GitHubRateLimitError` with specific error message
- [ ] 🟢 **GREEN** Catch `GitHubAuthenticationError` with specific error message
- [ ] 🟢 **GREEN** Catch `GitHubNotFoundError` with specific error message
- [ ] 🟢 **GREEN** Add generic exception catch as fallback
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Integration Test

- [ ] 🔴 **RED** Write integration test for full tool execution flow
- [ ] 🟢 **GREEN** Run integration tests for tool execution

---

### 2.4 MCP Configuration (TDD Cycle)

#### Feature: MCP Settings

- [ ] 🔴 **RED** Write test for `MCPSettings` loads from environment variables
- [ ] 🔴 **RED** Write parameterized test for default values
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `MCPSettings` class to `app/core/config.py`
- [ ] 🟢 **GREEN** Add `github_server_enabled` field
- [ ] 🟢 **GREEN** Add `github_server_port` field
- [ ] 🟢 **GREEN** Add `github_max_tools` field
- [ ] 🟢 **GREEN** Run tests, verify all pass
- [ ] ⚙️ **SETUP** Update `.env.example` with MCP configuration

---

### 2.5 MCP Server Testing (TDD Cycle)

#### Feature: Comprehensive MCP Test Coverage

- [ ] 🔴 **RED** Create `tests/mcp/test_github_server.py` file
- [ ] 🔴 **RED** Write test fixtures for MCP server
- [ ] 🔴 **RED** Write parameterized test for `list_tools` returns all expected tools
- [ ] 🔴 **RED** Write parameterized test for `call_tool` with valid arguments for each tool
- [ ] 🔴 **RED** Write parameterized test for `call_tool` with invalid tool names
- [ ] 🔴 **RED** Write parameterized test for `call_tool` with invalid arguments for each tool
- [ ] 🔴 **RED** Write parameterized test for error handling in each tool call
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement all MCP server components
- [ ] 🟢 **GREEN** Run tests, verify all pass
- [ ] 🔵 **REFACTOR** Review code for simplification
- [ ] 🔵 **REFACTOR** Verify test coverage > 85% for MCP server
- [ ] 🔵 **REFACTOR** Run tests after refactoring

---

## Phase 3: GitHub Agent (Week 3-4)

### 3.1 Agent Foundation (TDD Cycle)

#### Feature: Agent Initialization

- [ ] 🔴 **RED** Write test for `GitHubAgent` initializes with LLM and MCP server
- [ ] 🔴 **RED** Write test for `GitHubAgent` initializes with default_repo parameter
- [ ] 🔴 **RED** Write test for `GitHubAgent` initializes Pydantic output parser
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] ⚙️ **SETUP** Create `app/agents/github_agent.py` file
- [ ] ⚙️ **SETUP** Import required dependencies (langchain, pydantic, etc.)
- [ ] 🟢 **GREEN** Create `GitHubAgent` class skeleton
- [ ] 🟢 **GREEN** Implement `__init__` method with LLM and MCP server dependencies
- [ ] 🟢 **GREEN** Add `default_repo` parameter to `__init__`
- [ ] 🟢 **GREEN** Initialize Pydantic output parser for `GitHubIntent`
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 3.2 Intent Classification (TDD Cycle)

#### Feature: Intent Classification

- [ ] 🔴 **RED** Create test fixtures file `tests/fixtures/github_intent_fixtures.py`
- [ ] 🔴 **RED** Add 50+ example queries with expected intents to fixtures
- [ ] 🔴 **RED** Write parameterized test for `classify_intent` with various queries from fixtures
- [ ] 🔴 **RED** Write test for `classify_intent` handles parsing errors gracefully
- [ ] 🔴 **RED** Write test for `classify_intent` returns `GitHubIntent` with confidence > 0.7
- [ ] 🔴 **RED** Verify accuracy target: > 90% on test dataset
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Create `INTENT_PROMPT` ChatPromptTemplate
- [ ] 🟢 **GREEN** Define intent categories in system prompt
- [ ] 🟢 **GREEN** Add entity extraction instructions to system prompt
- [ ] 🟢 **GREEN** Add format instructions placeholder
- [ ] 🟢 **GREEN** Implement `classify_intent` async method
- [ ] 🟢 **GREEN** Add prompt formatting with user query
- [ ] 🟢 **GREEN** Add LLM invocation
- [ ] 🟢 **GREEN** Add response parsing with error handling
- [ ] 🟢 **GREEN** Add fallback for parsing errors
- [ ] 🟢 **GREEN** Run tests, verify all pass
- [ ] 🔵 **REFACTOR** Adjust prompt if accuracy < 90%
- [ ] 🔵 **REFACTOR** Add few-shot examples if needed
- [ ] 🔵 **REFACTOR** Run tests after refactoring

---

### 3.3 Query Routing (TDD Cycle)

#### Feature: Process Query Routing

- [ ] 🔴 **RED** Write parameterized test for `process_query` with valid GitHub queries
- [ ] 🔴 **RED** Write parameterized test for `process_query` with missing repo returns error
- [ ] 🔴 **RED** Write parameterized test for `process_query` uses default repo when not in context
- [ ] 🔴 **RED** Write parameterized test for `process_query` routes to correct MCP tool based on intent
- [ ] 🔴 **RED** Write test for `process_query` returns `AgentResponse` with success=True
- [ ] 🔴 **RED** Write test for `process_query` returns `AgentResponse` with success=False on error
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `process_query` async method
- [ ] 🟢 **GREEN** Add intent classification step
- [ ] 🟢 **GREEN** Add context extraction (repo, branch, etc.)
- [ ] 🟢 **GREEN** Add default repo fallback logic
- [ ] 🟢 **GREEN** Add error handling for missing repo
- [ ] 🟢 **GREEN** Implement `_route_to_tool` private method
- [ ] 🟢 **GREEN** Add routing logic for `issue:list` intent
- [ ] 🟢 **GREEN** Add routing logic for `issue:create` intent
- [ ] 🟢 **GREEN** Add routing logic for `issue:update` intent
- [ ] 🟢 **GREEN** Add routing logic for `issue:close` intent
- [ ] 🟢 **GREEN** Add `ValueError` for unsupported intents
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 3.4 Response Formatting (TDD Cycle)

#### Feature: Response Formatting

- [ ] 🔴 **RED** Write parameterized test for `_format_response` with `issue:list` results
- [ ] 🔴 **RED** Write test for `_format_response` limits to 10 issues
- [ ] 🔴 **RED** Write test for `_format_response` with `issue:create` results
- [ ] 🔴 **RED** Write test for `_format_response` with `issue:update` results
- [ ] 🔴 **RED** Write test for `_format_response` with `issue:close` results
- [ ] 🔴 **RED** Write test for `_format_response` with empty results shows "No issues found"
- [ ] 🔴 **RED** Write test for response includes issue count
- [ ] 🔴 **RED** Write test for responses are conversational (human-readable)
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `_format_response` method
- [ ] 🟢 **GREEN** Add formatter for `issue:list` results
- [ ] 🟢 **GREEN** Add limit to 10 issues to avoid long responses
- [ ] 🟢 **GREEN** Add formatter for `issue:create` results
- [ ] 🟢 **GREEN** Add formatter for `issue:update` results
- [ ] 🟢 **GREEN** Add formatter for `issue:close` results
- [ ] 🟢 **GREEN** Add "No issues found" message for empty results
- [ ] 🟢 **GREEN** Add issue count to response header
- [ ] 🟢 **GREEN** Make responses conversational and user-friendly
- [ ] 🟢 **GREEN** Run tests, verify all pass
- [ ] 🔵 **REFACTOR** Extract formatters to separate methods if complex
- [ ] 🔵 **REFACTOR** Run tests after refactoring

---

### 3.5 Error Handling (TDD Cycle)

#### Feature: Agent Error Handling

- [ ] 🔴 **RED** Write parameterized test for `process_query` with intent classification errors
- [ ] 🔴 **RED** Write parameterized test for `process_query` with tool execution errors
- [ ] 🔴 **RED** Write parameterized test for `process_query` with context extraction errors
- [ ] 🔴 **RED** Write test for errors return `AgentResponse` with success=False
- [ ] 🔴 **RED** Write test for errors have user-friendly messages
- [ ] 🔴 **RED** Write test for errors include details in metadata
- [ ] 🔴 **RED** Write test for errors are logged with full stack trace
- [ ] 🔴 **RED** Write test for errors don't crash the agent (agent continues to work)
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Wrap `process_query` in try-except block
- [ ] 🟢 **GREEN** Catch intent classification errors
- [ ] 🟢 **GREEN** Catch tool execution errors
- [ ] 🟢 **GREEN** Catch context extraction errors
- [ ] 🟢 **GREEN** Return `AgentResponse` with `success=False` on errors
- [ ] 🟢 **GREEN** Add user-friendly error messages
- [ ] 🟢 **GREEN** Add error details to metadata
- [ ] 🟢 **GREEN** Log all exceptions with full stack trace
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 3.6 Agent Testing (TDD Cycle)

#### Feature: Comprehensive Agent Test Coverage

- [ ] 🔴 **RED** Create `tests/agents/test_github_agent.py` file
- [ ] 🔴 **RED** Create test fixtures for LLM mocks
- [ ] 🔴 **RED** Create test fixtures for MCP server mocks
- [ ] 🔴 **RED** Write parameterized test for `classify_intent` with various queries
- [ ] 🔴 **RED** Write parameterized test for `process_query` with valid query
- [ ] 🔴 **RED** Write parameterized test for `process_query` with missing repo
- [ ] 🔴 **RED** Write parameterized test for `process_query` with unsupported intent
- [ ] 🔴 **RED** Write parameterized test for response formatting
- [ ] 🔴 **RED** Write parameterized test for error handling
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement all agent components
- [ ] 🟢 **GREEN** Run tests, verify all pass
- [ ] 🔵 **REFACTOR** Review code for simplification
- [ ] 🔵 **REFACTOR** Verify test coverage > 90%
- [ ] 🔵 **REFACTOR** Run tests after refactoring

---

## Phase 4: FinanceAgent Integration (Week 4-5)

### 4.1 Agent Routing (TDD Cycle)

#### Feature: GitHub Agent Routing

- [ ] 🔴 **RED** Write parameterized test for GitHub queries route to GitHubAgent
- [ ] 🔴 **RED** Write test for Jira queries still route to JiraAgent (no regression)
- [ ] 🔴 **RED** Write test for RAG queries still route to RAGAgent (no regression)
- [ ] 🔴 **RED** Write test for routing includes logging
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Open `app/agents/finance_agent.py`
- [ ] 🟢 **GREEN** Import `GitHubAgent` at top of file
- [ ] 🟢 **GREEN** Add `github_agent` parameter to `FinanceAgent.__init__`
- [ ] 🟢 **GREEN** Store `github_agent` as instance variable
- [ ] 🟢 **GREEN** Add GitHub keywords list to routing logic
- [ ] 🟢 **GREEN** Add routing condition for GitHub keywords
- [ ] 🟢 **GREEN** Add logging for GitHub routing
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 4.2 Dependency Injection (TDD Cycle)

#### Feature: DI Container Registration

- [ ] 🔴 **RED** Write test for `GitHubClient` is registered in DI container
- [ ] 🔴 **RED** Write test for `GitHubMCPServer` is registered in DI container
- [ ] 🔴 **RED** Write test for `GitHubAgent` is registered in DI container
- [ ] 🔴 **RED** Write test for `FinanceAgent` receives `github_agent` dependency
- [ ] 🔴 **RED** Write test for all dependencies resolve correctly
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Identify DI container location
- [ ] 🟢 **GREEN** Add `GitHubClient` registration to DI container
- [ ] 🟢 **GREEN** Add `GitHubMCPServer` registration to DI container
- [ ] 🟢 **GREEN** Add `GitHubAgent` registration to DI container
- [ ] 🟢 **GREEN** Update `FinanceAgent` factory to include `github_agent`
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 4.3 API Endpoints (TDD Cycle)

#### Feature: GitHub Query Endpoint

- [ ] 🔴 **RED** Create test file `tests/api/test_github_endpoints.py`
- [ ] 🔴 **RED** Write parameterized test for `POST /api/v1/github/query` with valid request
- [ ] 🔴 **RED** Write test for endpoint returns 202 with task_id
- [ ] 🔴 **RED** Write test for endpoint validates request with Pydantic models
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] ⚙️ **SETUP** Create `app/api/v1/github.py` file
- [ ] ⚙️ **SETUP** Create GitHub router
- [ ] 🟢 **GREEN** Implement `POST /api/v1/github/query` endpoint (sync)
- [ ] 🟢 **GREEN** Add Pydantic request models for endpoint
- [ ] 🟢 **GREEN** Add Pydantic response models for endpoint
- [ ] 🟢 **GREEN** Add error handling to endpoint
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: GitHub Chat Endpoint

- [ ] 🔴 **RED** Write parameterized test for `POST /api/v1/github/chat` with valid request
- [ ] 🔴 **RED** Write test for endpoint returns 202 with task_id
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `POST /api/v1/github/chat` endpoint (async)
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: GitHub Status Endpoint

- [ ] 🔴 **RED** Write parameterized test for `GET /api/v1/github/repo/{owner}/{repo}/status`
- [ ] 🔴 **RED** Write test for endpoint returns repository status
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `GET /api/v1/github/repo/{owner}/{repo}/status` endpoint
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: GitHub Issues Endpoints

- [ ] 🔴 **RED** Write parameterized test for `GET /api/v1/github/issues/{owner}/{repo}`
- [ ] 🔴 **RED** Write parameterized test for `POST /api/v1/github/issues/{owner}/{repo}`
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement `GET /api/v1/github/issues/{owner}/{repo}` endpoint
- [ ] 🟢 **GREEN** Implement `POST /api/v1/github/issues/{owner}/{repo}` endpoint
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: Router Integration

- [ ] 🔴 **RED** Write test for GitHub endpoints appear in OpenAPI docs
- [ ] 🔴 **RED** Write test for endpoints have request/response examples
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Open `app/api/v1/router.py` and include GitHub router
- [ ] 🟢 **GREEN** Verify endpoints appear in OpenAPI docs
- [ ] 🟢 **GREEN** Add request/response examples to OpenAPI docs
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 4.4 Celery Integration (TDD Cycle)

#### Feature: Celery Task Processing

- [ ] 🔴 **RED** Write test for Celery task recognizes GitHub queries
- [ ] 🔴 **RED** Write test for Celery task processes GitHub query with GitHubAgent
- [ ] 🔴 **RED** Write test for Celery task stores GitHub result correctly
- [ ] 🔴 **RED** Write test for Celery task handles GitHub errors gracefully
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Open `app/tasks/message_tasks.py`
- [ ] 🟢 **GREEN** Import GitHub agent and models
- [ ] 🟢 **GREEN** Add GitHub-specific task handling logic
- [ ] 🟢 **GREEN** Update task routing to recognize GitHub queries
- [ ] 🟢 **GREEN** Add GitHub result processing
- [ ] 🟢 **GREEN** Add error handling for GitHub tasks
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Integration Test

- [ ] 🔴 **RED** Write integration test for full async flow (API → Celery → Agent → GitHub)
- [ ] 🟢 **GREEN** Verify async processing works end-to-end

---

### 4.5 Integration Testing (TDD Cycle)

#### Feature: End-to-End Integration Tests

- [ ] 🔴 **RED** Create `tests/integration/test_github_flow.py` file
- [ ] 🔴 **RED** Write test for full chat flow (API → Celery → Agent → GitHub)
- [ ] 🔴 **RED** Write test for issue list flow
- [ ] 🔴 **RED** Write test for issue creation flow
- [ ] 🔴 **RED** Write test for error handling flow
- [ ] 🔴 **RED** Write test for concurrent requests
- [ ] 🔴 **RED** Write test for rate limiting behavior
- [ ] 🔴 **RED** Write test for existing agent tests still pass (no regressions)
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Implement all integration points
- [ ] 🟢 **GREEN** Run tests, verify all pass
- [ ] 🔵 **REFACTOR** Review integration tests for flakiness
- [ ] 🔵 **REFACTOR** Run tests after refactoring

---

## Phase 5: Advanced Features (Week 5-6)

### 5.1 Pull Request Operations (TDD Cycle)

#### Feature: PR Client Methods

- [ ] 🔴 **RED** Write parameterized test for `list_pull_requests` with various filters
- [ ] 🔴 **RED** Write parameterized test for `get_pull_request` returns PR details
- [ ] 🔴 **RED** Write parameterized test for `create_pr_comment` adds comment
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `GitHubPullRequestDetail` model to `app/models/github.py`
- [ ] 🟢 **GREEN** Add `list_pull_requests` method to `GitHubClient`
- [ ] 🟢 **GREEN** Add `get_pull_request` method to `GitHubClient`
- [ ] 🟢 **GREEN** Add `create_pr_comment` method to `GitHubClient`
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: PR MCP Tools

- [ ] 🔴 **RED** Write parameterized test for `github_list_prs` tool
- [ ] 🔴 **RED** Write parameterized test for `github_get_pr` tool
- [ ] 🔴 **RED** Write parameterized test for `github_create_pr_comment` tool
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `github_list_prs` tool to MCP server
- [ ] 🟢 **GREEN** Add `github_get_pr` tool to MCP server
- [ ] 🟢 **GREEN** Add `github_create_pr_comment` tool to MCP server
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: PR Agent Integration

- [ ] 🔴 **RED** Write parameterized test for PR intent classification
- [ ] 🔴 **RED** Write parameterized test for PR query routing
- [ ] 🔴 **RED** Write parameterized test for PR response formatting
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add PR intent handlers to `GitHubAgent`
- [ ] 🟢 **GREEN** Add PR response formatters
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 5.2 Code Search (TDD Cycle)

#### Feature: Code Search Client

- [ ] 🔴 **RED** Write parameterized test for `search_code` with various queries
- [ ] 🔴 **RED** Write test for `search_code` returns search results
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `search_code` method to `GitHubClient`
- [ ] 🟢 **GREEN** Add query parameter handling
- [ ] 🟢 **GREEN** Add response parsing for search results
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: Search MCP Tool & Agent

- [ ] 🔴 **RED** Write parameterized test for `github_search_code` tool
- [ ] 🔴 **RED** Write parameterized test for search intent classification
- [ ] 🔴 **RED** Write parameterized test for search response formatting
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `github_search_code` tool to MCP server
- [ ] 🟢 **GREEN** Add search intent handler to `GitHubAgent`
- [ ] 🟢 **GREEN** Add search response formatter
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 5.3 Repository Status (TDD Cycle)

#### Feature: Repository Status Client

- [ ] 🔴 **RED** Write parameterized test for `get_repository_status` with various branches
- [ ] 🔴 **RED** Write test for `get_repository_status` returns branch info and status checks
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `get_repository_status` method to `GitHubClient`
- [ ] 🟢 **GREEN** Add branch status retrieval
- [ ] 🟢 **GREEN** Add commit information retrieval
- [ ] 🟢 **GREEN** Add status checks retrieval
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: Status MCP Tool & Agent

- [ ] 🔴 **RED** Write parameterized test for `github_repo_status` tool
- [ ] 🔴 **RED** Write parameterized test for repo status intent classification
- [ ] 🔴 **RED** Write parameterized test for repo status response formatting
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `github_repo_status` tool to MCP server
- [ ] 🟢 **GREEN** Add repo status intent handler to `GitHubAgent`
- [ ] 🟢 **GREEN** Add repo status response formatter
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 5.4 File Operations (TDD Cycle)

#### Feature: File Contents Client

- [ ] 🔴 **RED** Write parameterized test for `get_file_contents` with various paths
- [ ] 🔴 **RED** Write test for `get_file_contents` decodes base64 content
- [ ] 🔴 **RED** Write test for `get_file_contents` validates file path
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `get_file_contents` method to `GitHubClient`
- [ ] 🟢 **GREEN** Add path validation
- [ ] 🟢 **GREEN** Add content decoding (base64)
- [ ] 🟢 **GREEN** Run tests, verify all pass

#### Feature: File MCP Tool & Agent

- [ ] 🔴 **RED** Write parameterized test for `github_get_file` tool
- [ ] 🔴 **RED** Write test for file content truncation for large files
- [ ] 🔴 **RED** Write parameterized test for file intent classification
- [ ] 🔴 **RED** Write parameterized test for file response formatting
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `github_get_file` tool to MCP server
- [ ] 🟢 **GREEN** Add file intent handler to `GitHubAgent`
- [ ] 🟢 **GREEN** Add file response formatter with truncation for large files
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 5.5 Additional MCP Tools (TDD Cycle)

#### Feature: Additional GitHub Tools

- [ ] 🔴 **RED** Write parameterized test for `github_get_branches` tool
- [ ] 🔴 **RED** Write parameterized test for `github_get_commits` tool
- [ ] 🔴 **RED** Write parameterized test for `github_get_user` tool
- [ ] 🔴 **RED** Verify tests fail (run pytest, expect failures)
- [ ] 🟢 **GREEN** Add `github_get_branches` tool
- [ ] 🟢 **GREEN** Add `github_get_commits` tool
- [ ] 🟢 **GREEN** Add `github_get_user` tool
- [ ] 🟢 **GREEN** Run tests, verify all pass

---

### 5.6 Comprehensive Testing (TDD Cycle)

#### Feature: Full Test Coverage

- [ ] 🔴 **RED** Run full test suite to identify gaps
- [ ] 🔴 **RED** Verify test coverage > 95%
- [ ] 🔴 **RED** Identify missing tests
- [ ] 🔴 **RED** Write missing tests to reach coverage target
- [ ] 🟢 **GREEN** Run integration tests with real GitHub API
- [ ] 🟢 **GREEN** Test rate limiting behavior
- [ ] 🟢 **GREEN** Test error handling for all error conditions
- [ ] 🟢 **GREEN** Fix any failing tests
- [ ] 🟢 **GREEN** Add missing tests to reach coverage target
- [ ] 🔵 **REFACTOR** Review test suite for quality
- [ ] 🔵 **REFACTOR** Remove duplicate tests
- [ ] 🔵 **REFACTOR** Improve test fixtures
- [ ] 🔵 **REFACTOR** Run tests after refactoring

---

## Phase 6: Documentation & Hardening (Week 6-7)

### 6.1 API Documentation

- [ ] ⚙️ **SETUP** Add docstrings to all `GitHubClient` methods (Google style)
- [ ] ⚙️ **SETUP** Add docstrings to all `GitHubMCPServer` methods (Google style)
- [ ] ⚙️ **SETUP** Add docstrings to all `GitHubAgent` methods (Google style)
- [ ] ⚙️ **SETUP** Verify all docstrings follow consistent style
- [ ] ⚙️ **SETUP** Add type hints to all public methods
- [ ] ⚙️ **SETUP** Verify OpenAPI docs include GitHub endpoints
- [ ] ⚙️ **SETUP** Add request/response examples to OpenAPI docs
- [ ] ⚙️ **SETUP** Generate API documentation

### 6.2 User Documentation

- [ ] ⚙️ **SETUP** Create user guide for GitHub commands
- [ ] ⚙️ **SETUP** Document example queries for each intent type
- [ ] ⚙️ **SETUP** Document supported GitHub operations
- [ ] ⚙️ **SETUP** Document configuration options
- [ ] ⚙️ **SETUP** Document rate limiting behavior
- [ ] ⚙️ **SETUP** Document error messages and their meanings
- [ ] ⚙️ **SETUP** Add troubleshooting section
- [ ] ⚙️ **SETUP** Review documentation for clarity

### 6.3 Performance Testing

- [ ] 🔴 **RED** Set up load testing framework (e.g., locust)
- [ ] 🔴 **RED** Write performance test: p95 latency < 5 seconds
- [ ] 🔴 **RED** Write performance test: p99 latency < 10 seconds
- [ ] 🔴 **RED** Write performance test: handle 100 concurrent users
- [ ] 🔴 **RED** Run load tests and verify benchmarks (expect failures if not met)
- [ ] 🟢 **GREEN** Create load test for GitHub queries
- [ ] 🟢 **GREEN** Run load test with 10 concurrent users
- [ ] 🟢 **GREEN** Run load test with 50 concurrent users
- [ ] 🟢 **GREEN** Run load test with 100 concurrent users
- [ ] 🟢 **GREEN** Measure p50, p95, p99 latencies
- [ ] 🟢 **GREEN** Verify p95 latency < 5 seconds
- [ ] 🔵 **REFACTOR** Identify and fix any performance bottlenecks
- [ ] 🔵 **REFACTOR** Re-run load tests after optimizations

### 6.4 Security Audit

- [ ] 🔴 **RED** Write security test: token never appears in logs
- [ ] 🔴 **RED** Write security test: token never appears in error messages
- [ ] 🔴 **RED** Write security test: API input validation prevents injection
- [ ] 🔴 **RED** Run security tests (expect failures if vulnerabilities found)
- [ ] 🟢 **GREEN** Review token storage and handling
- [ ] 🟢 **GREEN** Verify tokens never appear in logs
- [ ] 🟢 **GREEN** Verify tokens never appear in error messages
- [ ] 🟢 **GREEN** Review API input validation
- [ ] 🟢 **GREEN** Test for injection vulnerabilities
- [ ] 🟢 **GREEN** Review rate limiting implementation
- [ ] 🟢 **GREEN** Add audit logging for GitHub API calls
- [ ] 🟢 **GREEN** Create security review document

### 6.5 Monitoring Setup

- [ ] ⚙️ **SETUP** Add `github_agent_requests_total` metric
- [ ] ⚙️ **SETUP** Add `github_agent_errors_total` metric
- [ ] ⚙️ **SETUP** Add `github_agent_latency_seconds` histogram
- [ ] ⚙️ **SETUP** Add `github_api_rate_limit_remaining` gauge
- [ ] ⚙️ **SETUP** Add `github_mcp_tool_calls_total` counter (by tool name)
- [ ] ⚙️ **SETUP** Add `github_intent_classification_accuracy` gauge
- [ ] ⚙️ **SETUP** Create Grafana dashboard for GitHub metrics
- [ ] ⚙️ **SETUP** Add alerts for high error rate
- [ ] ⚙️ **SETUP** Add alerts for high latency
- [ ] ⚙️ **SETUP** Add alerts for low rate limit remaining

### 6.6 Deployment Preparation

- [ ] ⚙️ **SETUP** Create deployment guide
- [ ] ⚙️ **SETUP** Document environment variables
- [ ] ⚙️ **SETUP** Create GitHub PAT setup guide
- [ ] ⚙️ **SETUP** Document feature flag configuration
- [ ] ⚙️ **SETUP** Create rollback procedure document
- [ ] ⚙️ **SETUP** Test deployment in staging environment
- [ ] ⚙️ **SETUP** Verify all monitoring works in staging
- [ ] ⚙️ **SETUP** Create production deployment checklist

### 6.7 Final QA

- [ ] 🔴 **RED** Run full test suite (expect all pass)
- [ ] 🔴 **RED** Verify test coverage > 95%
- [ ] 🔴 **RED** Run mypy strict type checking (expect no errors)
- [ ] 🔴 **RED** Run linter (expect no errors)
- [ ] 🟢 **GREEN** Fix any failing tests
- [ ] 🟢 **GREEN** Add missing tests to reach coverage target
- [ ] 🟢 **GREEN** Fix any linting issues
- [ ] 🔵 **REFACTOR** Review all code changes
- [ ] 🔵 **REFACTOR** Create release notes
- [ ] 🔵 **REFACTOR** Tag release version
- [ ] 🔵 **REFACTOR** Merge to main branch

---

## Completion Checklist

### Pre-Merge (TDD Validation)

- [ ] All unit tests pass (run `pytest tests/`)
- [ ] All integration tests pass (run `pytest tests/integration/`)
- [ ] Test coverage > 95% (run `pytest --cov=app`)
- [ ] Code passes linter (run `ruff check` or `black --check`)
- [ ] Code passes type checker `mypy strict` (run `mypy app/`)
- [ ] All features follow TDD cycle (Red-Green-Refactor)
- [ ] All tests use parameterized testing where applicable
- [ ] Integration tests prioritized over excessive mocking
- [ ] Documentation complete
- [ ] Security review passed
- [ ] Performance benchmarks met
- [ ] No breaking changes to existing agents
- [ ] Feature flag implementation ready

### Pre-Production

- [ ] Staging deployment successful
- [ ] Load tests completed
- [ ] Monitoring configured
- [ ] Alerts configured
- [ ] Rollback plan documented
- [ ] On-call runbook created
- [ ] Stakeholder sign-off obtained

---

## Task Summary by Phase

| Phase | TDD Tasks | Setup Tasks | Total | Status |
|-------|-----------|-------------|-------|--------|
| Phase 1: Foundation | 49 | 6 | 55 | 🔴 Not Started |
| Phase 2: MCP Server | 38 | 4 | 42 | 🔴 Not Started |
| Phase 3: GitHub Agent | 42 | 0 | 42 | 🔴 Not Started |
| Phase 4: Integration | 30 | 0 | 30 | 🔴 Not Started |
| Phase 5: Advanced Features | 44 | 0 | 44 | 🔴 Not Started |
| Phase 6: Documentation | 10 | 35 | 45 | 🔴 Not Started |
| **Total** | **213** | **45** | **258** | **0% Complete** |

---

## Constitution Compliance Checklist

Each task implicitly follows:

- ✅ **Article 1: Simplicity First** - Only implement what's in spec.md
- ✅ **Article 2: Test-First** - All feature tasks start with 🔴 RED (write failing test)
- ✅ **Article 3: Clarity** - All public APIs have type hints and docstrings
- ✅ **Article 4: Single Responsibility** - Each component has one clear purpose

---

## Notes

- **🔴 RED tasks must be completed before 🟢 GREEN tasks** - This is non-negotiable per Constitution
- **Never skip writing the failing test first** - The test must fail before implementation
- **Use `@pytest.mark.parametrize`** for testing multiple inputs/edge cases
- **Prioritize integration tests** - Use real/mocked services rather than mocking internals
- **Run tests frequently** - After each GREEN phase, verify tests pass
- **Refactor only when tests are green** - Never refactor without test coverage
- **Update task checklist** - Mark tasks complete as you finish them
- **Raise blockers immediately** - If stuck, don't spin wheels alone

---

## Quick Reference: TDD Workflow Example

```bash
# 1. 🔴 RED: Write failing test
cat > tests/test_example.py << 'EOF'
import pytest

@pytest.mark.parametrize("input,expected", [
    ("open", "open"),
    ("closed", "closed"),
])
def test_issue_state_parsing(input, expected):
    """Test issue state parsing."""
    result = parse_issue_state(input)
    assert result == expected
EOF

# 2. 🔴 RED: Verify test fails
pytest tests/test_example.py -v  # Expected: FAIL

# 3. 🟢 GREEN: Implement minimal code to pass
cat > app/impl.py << 'EOF'
def parse_issue_state(state: str) -> str:
    return state
EOF

# 4. 🟢 GREEN: Verify test passes
pytest tests/test_example.py -v  # Expected: PASS

# 5. 🔵 REFACTOR: Improve while tests pass
# ... refactor code ...

# 6. 🔵 REFACTOR: Verify tests still pass
pytest tests/test_example.py -v  # Expected: PASS
```

---

**Document Status:** 🟢 Ready for TDD Implementation
**Constitution:** Article 2 (Test-First) Compliance: ✅
**Suggested Start Date:** TBD
**Target Completion Date:** TBD
