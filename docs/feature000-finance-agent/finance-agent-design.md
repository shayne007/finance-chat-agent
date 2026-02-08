# Finance Chat Agent: System Design Document

## 1. Overview

### 1.1 System Purpose
The Finance Chat Agent is a backend service that leverages GPT-4 to provide intelligent chat capabilities. It combines natural language processing with financial tools to assist users with their queries.

### 1.2 Key Architecture Decisions
- **Synchronous Architecture**: Direct API response model using FastAPI
- **Stateful Agent Workflows**: LangGraph for managing conversation state
- **Simple Tech Stack**: Python, FastAPI, LangChain/LangGraph, OpenAI

## 2. System Architecture

```mermaid
graph TB
    subgraph "External Systems"
        OPENAI[OpenAI GPT-4]
    end
    
    subgraph "API Layer"
        API[FastAPI Application]
        CHAT_API[/"POST /chat<br/>Chat Request"/]
    end
    
    subgraph "Processing Layer"
        AGENT[Finance Agent]
        GRAPH[LangGraph Workflow]
    end
    
    subgraph "Data Layer"
        MEMORY[In-Memory/Local Storage]
    end
    
    API --> AGENT
    AGENT --> GRAPH
    GRAPH --> OPENAI
    GRAPH --> MEMORY
```

## 3. Core Components

### 3.1 API Layer Design

```python
# app/api/endpoints.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
from datetime import datetime

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: str
    metadata: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    metadata: Optional[Dict[str, Any]] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Synchronous endpoint to handle chat requests
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Initialize agent
        agent = FinanceAgent()
        
        # Process message directly
        result = await agent.process_message(
            message=request.message,
            session_id=session_id,
            user_id=request.user_id,
            context=request.metadata
        )
        
        return ChatResponse(
            response=result["content"],
            session_id=session_id,
            metadata=result.get("metadata")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3.2 Finance Agent Core

```python
# app/agent/finance_agent.py
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class AgentState(TypedDict):
    """
    State definition for the Finance Agent workflow
    """
    messages: Annotated[List[BaseMessage], operator.add]
    user_id: str
    session_id: str
    context: dict

class FinanceAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        self.workflow = self._build_workflow()
        
    def _build_workflow(self):
        """Build the simple LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self.call_model)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_edge("agent", END)
        
        return workflow.compile()
    
    async def call_model(self, state: AgentState):
        """Process message with LLM"""
        messages = state["messages"]
        response = await self.llm.ainvoke(messages)
        return {"messages": [response]}

    async def process_message(self, message: str, session_id: str, user_id: str, context: dict = None):
        """Main entry point for processing messages"""
        
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "session_id": session_id,
            "context": context or {}
        }
        
        # Execute workflow
        result = await self.workflow.ainvoke(initial_state)
        
        # Extract response
        last_message = result["messages"][-1]
        
        return {
            "content": last_message.content,
            "metadata": result.get("context")
        }
```


### 3.3 Dynamic LLM Management

```python
# app/llm/manager.py
import openai
from datetime import datetime, timedelta
from typing import Optional
from cachetools import TTLCache
import logging

class OpenAIManager:
    """
    Manages OpenAI LLM instances with dynamic API key management
    """
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.api_key_cache = TTLCache(maxsize=100, ttl=300)  # 5-minute cache
        self.logger = logging.getLogger(__name__)
        
    def _get_valid_api_key(self) -> str:
        """
        Get a valid API key, refreshing if expired
        """
        user_id = self.kwargs.get("user_id", "default")
        cached_key = self.api_key_cache.get(user_id)
        
        if cached_key and not self._is_key_expired(cached_key):
            return cached_key["key"]
        
        # Fetch new key from database or config service
        new_key = self._refresh_api_key(user_id)
        self.api_key_cache[user_id] = {
            "key": new_key,
            "expires_at": datetime.utcnow() + timedelta(hours=1)
        }
        
        return new_key
    
    def _refresh_api_key(self, user_id: str) -> str:
        """
        Refresh the API key from the configuration service
        """
        from app.config.service import ConfigService
        
        config_service = ConfigService()
        api_config = config_service.get_openai_config(user_id)
        
        if not api_config or not api_config.get("api_key"):
            raise ValueError(f"No valid OpenAI API key found for user {user_id}")
        
        return api_config["api_key"]
    
    def _is_key_expired(self, key_data: dict) -> bool:
        """Check if API key has expired"""
        expires_at = key_data.get("expires_at")
        if not expires_at:
            return True
        
        return datetime.utcnow() > expires_at
    
    def invoke(self, messages, **invoke_kwargs):
        """
        Invoke the LLM with automatic API key management
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                api_key = self._get_valid_api_key()
                
                client = openai.OpenAI(api_key=api_key)
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **{**self.kwargs, **invoke_kwargs}
                )
                
                return response.choices[0].message
                
            except openai.AuthenticationError as e:
                if attempt < max_retries - 1:
                    # Clear cache and retry
                    user_id = self.kwargs.get("user_id", "default")
                    self.api_key_cache.pop(user_id, None)
                    self.logger.warning(f"Authentication error, retrying... Attempt {attempt + 1}")
                    continue
                else:
                    raise
            except Exception as e:
                self.logger.error(f"LLM invocation failed: {str(e)}")
                raise
```

### 3.4 Celery Task Integration

```python
# app/tasks/celery_tasks.py
from celery import Celery, Task
from app.agent.graph import JiraAIAgent
from app.config.settings import settings
import json

celery_app = Celery(
    'jira_ai_agent',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=['app.tasks.celery_tasks']
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    broker_connection_retry_on_startup=True
)

class AgentTask(Task):
    """Base task class with agent initialization"""
    
    def __init__(self):
        super().__init__()
        self.agent = None
    
    def initialize_agent(self):
        if not self.agent:
            from app.agent.graph import JiraAIAgent
            from app.config.settings import settings
            
            self.agent = JiraAIAgent(settings)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        from app.storage.result_store import ResultStore
        
        result_store = ResultStore()
        result_store.update(
            task_id,
            {
                "status": "failed",
                "error": str(exc),
                "completed_at": datetime.utcnow().isoformat()
            }
        )

@celery_app.task(base=AgentTask, bind=True, name="process_chat_request")
def process_chat_request(self, task_id, session_id, message, user_id, metadata):
    """Process chat request asynchronously"""
    self.initialize_agent()
    
    # Initialize agent state
    initial_state = {
        "messages": [{"role": "user", "content": message}],
        "user_id": user_id,
        "session_id": session_id,
        "task_id": task_id,
        "jira_context": metadata.get("jira_context", {}),
        "format": metadata.get("format", "markdown"),
        "tool_outputs": [],
        "next_step": "analyze"
    }
    
    # Execute the graph
    try:
        config = {"configurable": {"thread_id": session_id}}
        
        for event in self.agent.graph.stream(initial_state, config):
            # Store intermediate results
            if "response" in event:
                from app.storage.result_store import ResultStore
                
                result_store = ResultStore()
                result_store.update(
                    task_id,
                    {
                        "status": "completed",
                        "response": event["response"],
                        "session_id": session_id,
                        "completed_at": datetime.utcnow().isoformat()
                    }
                )
                
    except InterruptionRequired as e:
        # Workflow paused for user input
        from app.storage.result_store import ResultStore
        
        result_store = ResultStore()
        result_store.update(
            task_id,
            {
                "status": "paused",
                "interruption_type": e.interruption_type,
                "required_data": e.data,
                "session_id": session_id,
                "paused_at": datetime.utcnow().isoformat()
            }
        )
        
### 3.4 Prompt Management System

```python
# app/prompts/manager.py
from typing import Dict, Any
import os
import yaml
from pathlib import Path
from jinja2 import Template

class PromptManager:
    """
    Manages prompts stored as markdown files with templating
    """
    
    def __init__(self, prompt_dir: str = "prompts"):
        self.prompt_dir = Path(prompt_dir)
        self.prompt_cache = {}
        self.load_all_prompts()
    
    def load_all_prompts(self):
        """Load all prompts from markdown files"""
        for prompt_file in self.prompt_dir.glob("*.md"):
            prompt_name = prompt_file.stem
            with open(prompt_file, 'r') as f:
                content = f.read()
            
            # Parse metadata and template
            prompt_data = self._parse_prompt_markdown(content)
            self.prompt_cache[prompt_name] = prompt_data
    
    def _parse_prompt_markdown(self, content: str) -> Dict[str, Any]:
        """
        Parse markdown file with YAML frontmatter
        Format:
        ---
        name: finance_advisor
        version: 1.0
        variables:
          - user_message
          - history
        ---
        
        # System Prompt
        
        {{ instruction }}
        """
        lines = content.split('\n')
        
        if lines[0] == '---':
            # Parse YAML frontmatter
            yaml_lines = []
            for line in lines[1:]:
                if line == '---':
                    break
                yaml_lines.append(line)
            
            metadata = yaml.safe_load('\n'.join(yaml_lines))
            
            # Extract template content
            template_start = len(yaml_lines) + 2
            template_content = '\n'.join(lines[template_start:])
            
            return {
                "metadata": metadata,
                "template": template_content
            }
        else:
            # No metadata, treat entire content as template
            return {
                "metadata": {"name": "unnamed"},
                "template": content
            }
    
    def get_prompt(self, prompt_name: str, version: str = None) -> "PromptTemplate":
        """Get a prompt template by name"""
        prompt_data = self.prompt_cache.get(prompt_name)
        if not prompt_data:
             raise ValueError(f"Prompt {prompt_name} not found")
        
        return PromptTemplate(
            name=prompt_name,
            template=prompt_data["template"],
            variables=prompt_data["metadata"].get("variables", [])
        )

class PromptTemplate:
    """Renders prompt templates with variables"""
    
    def __init__(self, name: str, template: str, variables: list):
        self.name = name
        self.template = Template(template)
        self.variables = variables
    
    def format_messages(self, **kwargs) -> list:
        """Format the prompt into OpenAI message format"""
        # Check all required variables are provided
        missing_vars = [var for var in self.variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing variables: {missing_vars}")
        
        # Render template
        rendered = self.template.render(**kwargs)
        
        # Split into system and user messages if marked
        if "## System:" in rendered and "## User:" in rendered:
            system_part, user_part = rendered.split("## User:")
            system_content = system_part.replace("## System:", "").strip()
            user_content = user_part.strip()
            
            return [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        else:
            # Assume it's a user message
            return [{"role": "user", "content": rendered.strip()}]
```

## 4. Workflow Diagram

```mermaid
stateDiagram-v2
    [*] --> ProcessMessage
    
    state ProcessMessage {
        [*] --> LoadContext
        LoadContext --> ConstructPrompt
        ConstructPrompt --> CallLLM
        CallLLM --> [*]
    }
    
    ProcessMessage --> GenerateResponse
    
    state GenerateResponse {
        [*] --> FormatOutput
        FormatOutput --> [*]
    }
    
    GenerateResponse --> [*]
```

## 5. Data Models

```python
# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class AgentResponse(BaseModel):
    """Schema for agent response"""
    session_id: str
    status: str = "completed"
    response: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PromptMetadata(BaseModel):
    """Schema for prompt metadata"""
    name: str
    version: str
    description: Optional[str]
    variables: List[str]
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    tags: List[str] = []
```

## 6. Use Cases with Examples

### 6.1 Use Case 1: General Financial Question

**User Input:**
```
"What is the difference between a stock and a bond?"
```

**Agent Workflow:**
1. **Receive Message**: API receives the user's question.
2. **Context Loading**: Agent loads previous conversation history (if any).
3. **LLM Processing**: GPT-4 processes the question with financial context.
4. **Response Generation**: Generates a clear, educational explanation.

**Output:**
```json
{
  "response": {
    "text": "A stock represents ownership in a company (equity), while a bond is a loan you give to a company or government (debt). Stocks offer higher potential returns but come with higher risk, whereas bonds provide regular interest payments and are generally safer."
  },
  "session_id": "uuid-1234",
  "status": "completed"
}
```

### 6.2 Use Case 2: Investment Advice Disclaimer

**User Input:**
```
"Should I buy Apple stock right now?"
```

**Agent Workflow:**
1. **Receive Message**: API receives the specific investment question.
2. **LLM Processing**: Model detects request for specific investment advice.
3. **Guardrails**: System prompt instructs to provide data but avoid specific financial advice.
4. **Response Generation**: Returns recent market data (if enabled) and a disclaimer.

**Output:**
```json
{
  "response": {
    "text": "I cannot provide personalized financial advice. However, Apple (AAPL) is currently trading at $150. Analysts often look at P/E ratios and recent earnings reports to evaluate value. You should consult a qualified financial advisor before making investment decisions."
  },
  "session_id": "uuid-5678",
  "status": "completed"
}
```
```python
# Agent detects need for confirmation
raise InterruptionRequired(
    interruption_type="confirmation_required",
    data={
        "message": "This will update 15 related tickets. Proceed?",
        "action": "bulk_update",
        "affected_tickets": ["TICK-1", "TICK-2", ...],
        "risk_level": "medium"
    }
)
```

**User Interaction via API:**
```bash
# Check paused task
GET /chat/task_123

# Response:
{
  "status": "paused",
  "interruption_type": "confirmation_required",
  "required_data": {
    "message": "This will update 15 related tickets. Proceed?",
    "action": "bulk_update"
  }
}

# Send confirmation
POST /chat/task_123/resume
{
  "user_input": "yes, proceed with changes",
  "parameters": {"notify_team": true}
}
```

## 7. Deployment Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/jira_ai
      - REDIS_URL=redis://redis:6379/0
      - JIRA_BASE_URL=${JIRA_BASE_URL}
      - JIRA_API_TOKEN=${JIRA_API_TOKEN}
    depends_on:
      - postgres
      - redis
      - celery_worker
  
  celery_worker:
    build: .
    command: celery -A app.tasks.celery_tasks worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/jira_ai
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=jira_ai
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## 8. Security Considerations

### 8.1 API Key Management
- **Rotation**: Automatic API key refresh based on expiration
- **Isolation**: User-specific API keys with rate limiting
- **Encryption**: Keys encrypted at rest and in transit

### 8.2 Data Protection
- **PII Handling**: LLM responses sanitized for sensitive information
- **Audit Logs**: All actions logged for compliance
- **Access Control**: Role-based access to Jira projects

### 8.3 Rate Limiting
```python
# app/middleware/rate_limit.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour", "10/minute"]
)

# Apply different limits based on user tier
def get_user_limit(user_id: str):
    user_tier = user_service.get_tier(user_id)
    
    limits = {
        "free": ["50/hour", "5/minute"],
        "pro": ["500/hour", "50/minute"],
        "enterprise": ["5000/hour", "500/minute"]
    }
    
    return limits.get(user_tier, ["100/hour"])
```

## 9. Monitoring and Observability

### 9.1 Metrics Collection
```python
# app/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Agent metrics
AGENT_REQUESTS = Counter('agent_requests_total', 'Total agent requests')
AGENT_REQUEST_DURATION = Histogram('agent_request_duration_seconds', 'Request duration')
AGENT_ERRORS = Counter('agent_errors_total', 'Agent errors by type', ['error_type'])
LLM_TOKEN_USAGE = Counter('llm_tokens_total', 'LLM token usage', ['model', 'type'])

# Jira metrics
JIRA_API_CALLS = Counter('jira_api_calls_total', 'Jira API calls', ['endpoint', 'status'])
JIRA_LATENCY = Histogram('jira_api_latency_seconds', 'Jira API latency')

# Workflow metrics
WORKFLOW_STATES = Gauge('workflow_states', 'Current workflow states', ['state'])
INTERRUPTIONS = Counter('workflow_interruptions_total', 'Workflow interruptions', ['type'])
```

### 9.2 Logging Configuration
```python
# app/logging/config.py
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Agent-specific logging
agent_logger = structlog.get_logger("jira_ai_agent")
agent_logger = agent_logger.bind(
    component="agent",
    version="1.0.0"
)
```

## 10. Scaling Considerations

### 10.1 Horizontal Scaling
- **Stateless API Layer**: API servers can be scaled independently
- **Celery Workers**: Worker pools can be scaled based on queue depth
- **Redis Cluster**: For high-volume task queuing
- **PostgreSQL Read Replicas**: For checkpoint and state storage

### 10.2 Caching Strategy
```python
# app/cache/redis_cache.py
from redis import RedisCluster
from functools import wraps
import pickle

class AgentCache:
    def __init__(self):
        self.redis = RedisCluster.from_url(
            settings.REDIS_CLUSTER_URL,
            decode_responses=False
        )
    
    def cached(self, ttl=300):
        """Decorator for caching agent responses"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                key = self._generate_key(func.__name__, args, kwargs)
                
                # Try cache
                cached = self.redis.get(key)
                if cached:
                    return pickle.loads(cached)
                
                # Execute and cache
                result = func(*args, **kwargs)
                self.redis.setex(key, ttl, pickle.dumps(result))
                return result
            return wrapper
        return decorator
```

## 11. Conclusion

The Jira AI Agent system provides a robust, scalable solution for integrating LLM capabilities with Jira ticket management. Key strengths include:

1. **Flexible Architecture**: Modular design with clear separation of concerns
2. **Human-in-the-Loop**: Support for interruptions and confirmations
3. **Scalability**: Async processing with Celery and horizontal scaling
4. **Maintainability**: Externalized prompts and configuration
5. **Observability**: Comprehensive monitoring and logging
6. **Security**: Proper API key management and data protection

The system effectively bridges natural language interaction with structured Jira workflows, providing an intuitive interface for both technical and non-technical users while maintaining the rigor and auditability required for enterprise use.

## 12. Appendices

### 12.1 Sample Prompt Templates

**intent_classification.md**
```markdown
---
name: intent_classification
version: 1.1
variables:
  - user_message
  - history
---

## System:
You are a Jira assistant that classifies user intents. Analyze the message and determine the intent.

## User:
Message: {{ user_message }}

Previous conversation:
{% for msg in history %}
{{ msg.role }}: {{ msg.content }}
{% endfor %}

## Instructions:
Classify the intent into one of:
- create_ticket: User wants to create a new Jira ticket
- assess_ticket: User wants to analyze or get info about a ticket
- analyze_requirements: User wants requirements analyzed
- search_tickets: User wants to search for tickets
- update_ticket: User wants to update an existing ticket
- unknown: Cannot determine intent

Respond with JSON only: {"intent": "<value>", "confidence": 0.95, "entities": {}}
```

**ticket_creation.md**
```markdown
---
name: ticket_creation
version: 1.2
variables:
  - requirement
  - context
---

## System:
You are a Jira ticket creation assistant. Extract structured data from natural language requirements.

## User:
Requirement: {{ requirement }}

Project Context: {{ context }}

## Instructions:
Extract the following fields:
1. Project key (if mentioned)
2. Issue type (Bug, Task, Story, Epic, Improvement)
3. Summary (concise, max 255 chars)
4. Detailed description
5. Priority (Highest, High, Medium, Low, Lowest)
6. Assignee (if specified)
7. Labels (relevant tags)

If any information is missing, note what needs to be clarified.

Respond with JSON only.
```

### 12.2 Error Handling Strategy

```python
# app/errors/handlers.py
class AgentError(Exception):
    """Base exception for agent errors"""
    pass

class InterruptionRequired(AgentError):
    """Raised when workflow needs user input"""
    def __init__(self, interruption_type, data=None):
        self.interruption_type = interruption_type
        self.data = data or {}
        super().__init__(f"Interruption required: {interruption_type}")

class LLMError(AgentError):
    """LLM-related errors"""
    pass

class JiraAPIError(AgentError):
    """Jira API errors"""
    pass

# Global error handler
@app.exception_handler(AgentError)
async def agent_error_handler(request, exc):
    if isinstance(exc, InterruptionRequired):
        return JSONResponse(
            status_code=202,  # Accepted, but needs input
            content={
                "error": "interruption_required",
                "type": exc.interruption_type,
                "data": exc.data
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "agent_error",
                "message": str(exc),
                "type": exc.__class__.__name__
            }
        )
```