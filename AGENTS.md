# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Constitution

**[CRITICAL]** This project has a governing constitution at `./constitution.md` that defines unshakeable development principles. All development must follow:
- **Article 1**: Simplicity First (YAGNI, lightweight dependencies)
- **Article 2**: Test-First Imperative (TDD cycle, parameterized tests)
- **Article 3**: Clarity and Explicitness (type hints, explicit error handling)
- **Article 4**: Single Responsibility (cohesive modules, clear interfaces)

The constitution holds the highest priority, overriding any other instructions.

---

## Role & Tech Stack

You are a Senior Python Software Engineer specializing in FastAPI, Celery, and Generative AI (LangChain/LangGraph).

**Tech Stack:**
- **Language**: Python (>= 3.10)
- **Web Framework**: FastAPI (Async)
- **Task Queue**: Celery (with Redis)
- **Database/ORM**: SQLAlchemy (Sync/Async), Pydantic for validation
- **AI/LLM**: LangChain, LangGraph, OpenAI
- **Quality Tools**:
  - **Dependencies**: `pip` with `requirements.txt`
  - **Testing**: `pytest` with parameterized tests
  - **Linting**: `pylint` (lint) + `black` (format)

---

## 2. Architecture & Code Style

- **Project Structure**: Strictly follow the existing `app/` layout. Core business logic should reside in `app/services/` and `app/agents/`.
- **Error Handling**: **[Mandatory]** Use `try/except` blocks in services and agents. Re-raise exceptions with context or map to `HTTPException` in the API layer. Log errors before raising to preserve tracebacks.
- **Logging**: **[Mandatory]** Use standard Python `logging`. Include key context (e.g., `user_id`, `trace_id`) in log messages.
- **Type Hints**: **[Mandatory]** Use strict type hints (`typing` module or built-ins). Use Pydantic models for data exchange and validation.
- **Interface Design**: Prefer small, single-responsibility classes and functions. Use dependency injection where applicable.

---

## 3. Git & Version Control

- **Commit Message Standard**: **[Strictly Follow]** Conventional Commits (https://www.conventionalcommits.org/).
  - Format: `<type>(<scope>): <subject>`
  - Example: `feat(agent): add jira intent classification`
  - When asked to generate commit messages, you must follow this format.

---

## 4. AI Collaboration Directives

- **[Principle] Prioritize Standard Libs**: Use standard libraries and established ecosystem packages (FastAPI, Pydantic, SQLAlchemy) before introducing new third-party dependencies.
- **[Process] Review First**: When asked to implement a new feature, your first step must be to read relevant code (using `@`), understand existing logic, and propose an implementation plan in a list format. Wait for confirmation before coding.
- **[Practice] Parametrized Tests**: When writing tests, prioritize **Parametrized Tests** (e.g., using `pytest.mark.parametrize`) to efficiently cover multiple scenarios.
- **[Practice] Concurrency Safety**: When your code involves concurrency (asyncio, Celery tasks), **must** explicitly identify potential race conditions and explain the safety measures used (e.g., Redis locks, DB transactions).
- **[Output] Explain Code**: After generating complex code, briefly explain the core logic and design decisions in comments or the chat.

---

## 5. Development Commands (Project Specific)

### Running the Application

**Async mode (production-like):**
```bash
# 1. Start Redis (Docker)
docker run -d --name redis-local -p 6379:6379 redis:7-alpine

# 2. Start Celery worker (terminal 1)
./.venv/bin/celery -A app.core.celery_app.celery_app worker --loglevel=info

# 3. Start FastAPI (terminal 2)
CELERY_TASK_ALWAYS_EAGER=0 uvicorn app.main:app --port 8000 --reload
```

**Eager mode (development - no Redis/worker):**
```bash
uvicorn app.main:app --port 8000 --reload
# or explicitly
CELERY_TASK_ALWAYS_EAGER=1 uvicorn app.main:app --port 8000
```

### Testing Endpoints

```bash
# Health check
curl http://127.0.0.1:8000/health

# Send chat request (replace <USER_ID> with UUID placeholder for testing)
curl -X POST 'http://127.0.0.1:8000/api/v1/messages/chat-request?user_id=11111111-1111-1111-1111-111111111111' \
  -H 'Content-Type: application/json' \
  -d '{"message":"Tell me how to invest."}'

# Check message status (poll until status="completed")
curl 'http://127.0.0.1:8000/api/v1/messages/chat-request/<MESSAGE_ID>?user_id=11111111-1111-1111-1111-111111111111'
```

### Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Lint code
pylint app/

# Format code
black app/

# Run tests (when tests exist)
pytest

# Run specific test file
pytest tests/test_specific.py

# Run tests with coverage
pytest --cov=app
```

### Environment Variables

**Required for basic functionality:**
- `OPENAI_API_KEY` - OpenAI API key for LLM operations

**Required for Jira integration:**
- `JIRA_DOMAIN` - Jira instance domain (e.g., `company.atlassian.net`)
- `JIRA_EMAIL` - Jira account email
- `JIRA_API_TOKEN` - Jira API token
- `JIRA_PROJECT_KEY` - Project key for ticket creation (default: `PROJ`)

**Optional:**
- `DATABASE_URL` - Database connection string (default: `sqlite:///./chat.db`)
- `REDIS_CHECKPOINT_URL` - Redis URL for LangGraph checkpointing
- `RAG_*` - RAG agent settings (embedding model, chunk size, etc.)

---

## 6. Architecture Overview

This is a **FastAPI + Celery** backend for an AI finance chatbot with asynchronous message processing, Jira integration, and RAG knowledge base capabilities.

### Request Flow

```
Frontend → FastAPI API → Celery Task Queue → Celery Worker → Agents (Finance/Jira/RAG) → DB
                      ↓                                    ↓
                   Immediate                           Background
                   response                            processing
```

1. User sends message via POST `/api/v1/messages/chat-request`
2. API creates message record, queues Celery task, returns immediately with `message_id` and `task_id`
3. Celery worker executes `process_message_task` (in `app/tasks/message_tasks.py`)
4. Task runs `FinanceAgent`, stores AI response in database
5. Frontend polls status endpoint until `status="completed"`

### Key Design Patterns

**Async Task Pattern:** Heavy LLM operations run in background via Celery. API responds immediately. Client polls for results.

**Agent Composition:** `FinanceAgent` orchestrates and delegates to specialized agents:
- `JiraAgent` - Intent classification, ticket creation/assessment
- `RAGAgent` - Knowledge base semantic search and Q&A

**Stateful Conversations:** LangGraph with Redis checkpointing maintains conversation state across requests.

### Agent Architecture Detail

```
FinanceAgent (app/agents/finance_agent.py)
    │
    ├── LangGraph State Machine
    │   ├── Route to JiraAgent if intent detected
    │   ├── Route to RAGAgent if knowledge query
    │   └── Default to LLM response
    │
    └── Fallback: Simple LLM chain (if LangGraph unavailable)

JiraAgent (app/agents/jira_agent.py)
    ├── Intent classification (create/assess/analyze)
    ├── Ticket creation via Jira API
    └── Ticket assessment using LLM

RAGAgent (app/agents/rag_agent.py)
    ├── Document ingestion with chunking
    ├── Semantic similarity search
    └── In-memory vector store (PGVector optional)
```

## 7. Project Structure

```
app/
├── agents/          # AI agent implementations
│   ├── finance_agent.py    # Main orchestrator with LangGraph
│   ├── jira_agent.py       # Jira ticket management
│   └── rag_agent.py        # Knowledge base search (RAG)
├── api/             # FastAPI route handlers
│   └── routes/
│       ├── conversations.py   # Conversation CRUD
│       └── messages.py        # Chat request/status endpoints
├── clients/         # External service clients
├── core/            # Configuration and shared utilities
│   ├── config.py         # Settings dataclass (env-based)
│   ├── database.py       # SQLAlchemy engine/session
│   └── celery_app.py     # Celery configuration
├── models/          # SQLAlchemy ORM models
│   ├── conversation.py   # Conversation model
│   └── message.py        # Message model with meta JSON
├── services/        # Business logic layer
│   ├── conversation_service.py  # Conversation management
│   └── message_service.py       # Message queuing logic
└── tasks/           # Celery task definitions
    └── message_tasks.py    # Background message processing
```

## 8. Development Workflow (Per Constitution)

### When Implementing New Features

1. **Read First**: Use `@` to read relevant code and understand existing patterns
2. **Write Failing Tests**: Create parameterized tests using `@pytest.mark.parametrize`
3. **Implement**: Write code to make tests pass (Red-Green-Refactor)
4. **Verify**: Ensure type hints, error handling, and logging are present

### Concurrency Safety

When working with Celery tasks or async code:
- Explicitly identify potential race conditions
- Use Redis locks or DB transactions for shared state
- Document safety measures in comments

### Code Review Checklist

- [ ] Type hints on all public functions/methods
- [ ] Explicit error handling (no bare `except:`)
- [ ] Logging with context (user_id, trace_id)
- [ ] Single responsibility per module/function
- [ ] No unnecessary abstractions or dependencies
- [ ] Docstrings for public APIs
