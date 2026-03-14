# finance-chat-agent

FastAPI backend with Celery-powered asynchronous message processing for a finance chat agent.

## Requirements

- Python 3.9+
- Virtualenv (recommended)
- Docker Desktop (for Redis), or any Redis server reachable at `redis://localhost:6379/0`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start Services

### Option A: Async mode (recommended)

1) Start Redis (Docker):

```bash
docker run -d --name redis-local -p 6379:6379 redis:7-alpine
```

2) Start Celery worker (in a terminal):

```bash
./.venv/bin/celery -A app.core.celery_app.celery_app worker --loglevel=info
```

3) Start FastAPI (in another terminal):

```bash
CELERY_TASK_ALWAYS_EAGER=0 uvicorn app.main:app --port 8000
```

Environment variables (override defaults if needed):

```bash
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

## Test Endpoints

Replace `<USER_ID>` with your UUID (for local testing a placeholder works): `11111111-1111-1111-1111-111111111111`.

### 1) Enqueue a chat request

```bash
curl -sS -X POST \
  'http://127.0.0.1:8000/api/v1/messages/chat-request?user_id=11111111-1111-1111-1111-111111111111' \
  -H 'Content-Type: application/json' \
  -d '{"message":"Tell me how to invest."}'
```

Example response:

```json
{
  "message_id": "<MESSAGE_ID>",
  "task_id": "<TASK_ID>",
  "status": "queued",
  "response_message_id": null,
  "content": null,
  "conversation_id": null,
  "created_at": null
}
```

### 2) Check chat-request status (poll from UI)

```bash
curl -sS \
  'http://127.0.0.1:8000/api/v1/messages/chat-request/2d467efb-d304-4805-a3fc-d07eb7a485e9?user_id=11111111-1111-1111-1111-111111111111'
```

When processing:

```json
{
  "message_id": "<MESSAGE_ID>",
  "task_id": "<TASK_ID>",
  "status": "queued"
}
```

When completed (includes the LLM answer):

```json
{
  "message_id": "<MESSAGE_ID>",
  "task_id": "<TASK_ID>",
  "status": "completed",
  "response_message_id": "<ASSISTANT_MESSAGE_ID>",
  "content": "<LLM_ANSWER>",
  "conversation_id": "<CONVERSATION_ID>",
  "created_at": "<ISO_TIMESTAMP>"
}
```

### 3) Health check

```bash
curl -sS 'http://127.0.0.1:8000/health'
```

## Skills-based Agent with SkillMiddleware

The finance chat agent now supports **SkillMiddleware** - a progressive disclosure pattern that reduces token consumption by 60-80% while maintaining full capabilities.

### Features

- **Progressive Disclosure**: Only load skill metadata upfront, load full content on-demand
- **Token Efficiency**: 60-80% token reduction through on-demand skill loading
- **Team Autonomy**: Different teams can maintain specialized skills independently
- **Scalable**: Add dozens of skills without overwhelming context

### Skills API Endpoints

```bash
# Get available skills information
curl -sS 'http://127.0.0.1:8000/api/v1/skills/info'

# Analyze code with skills
curl -X POST 'http://127.0.0.1:8000/api/v1/skills/analyze' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Analyze this Python code and identify classes and functions",
    "code": "class Calculator:\n    def add(self, a, b):\n        return a + b"
  }'

# Generate documentation with skills
curl -X POST 'http://127.0.0.1:8000/api/v1/skills/generate-documentation' \
  -H 'Content-Type: application/json' \
  -d '{
    "repo_path": "./my-project",
    "format": "markdown",
    "requirements": "Generate comprehensive documentation with API endpoints"
  }'

# Create architecture diagrams
curl -X POST 'http://127.0.0.1:8000/api/v1/skills/create-architecture-diagram' \
  -H 'Content-Type: application/json' \
  -d '{
    "description": "Microservices architecture with API Gateway",
    "diagram_type": "architecture"
  }'
```

### Running Skills Examples

```bash
# Run the skills agent example
python examples/skills_agent_example.py

# Run tests
python -m pytest tests/test_skills_middleware.py -v
```

## Notes

- In async mode, ensure Redis is reachable and the Celery worker is running.
- The UI should implement the polling loop client-side, calling the status endpoint until `status` is `completed`.
- Conversation messages can be fetched via `GET /api/v1/messages/{conversation_id}` if needed for full history.
- Skills are loaded on-demand and cached for better performance. Check the response metadata to see which skills were used.