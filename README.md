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

## Notes

- In async mode, ensure Redis is reachable and the Celery worker is running.
- The UI should implement the polling loop client-side, calling the status endpoint until `status` is `completed`.
- Conversation messages can be fetched via `GET /api/v1/messages/{conversation_id}` if needed for full history.