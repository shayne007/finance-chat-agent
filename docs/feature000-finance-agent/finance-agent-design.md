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
        GITHUB_API[GitHub API]
        JIRA_API[Jira API]
    end
    
    subgraph "API Layer"
        API[FastAPI Application]
        CHAT_API[/"POST /chat<br/>Chat Request"/]
    end
    
    subgraph "Finance Agent Router"
        ROUTER[Intent Classifier]
        CHAT[Chat Agent]
        JIRA[Jira Agent]
        GITHUB[GitHub Agent]
        RAG[RAG Agent]
    end
    
    subgraph "Data Layer"
        MEMORY[In-Memory/Local Storage]
    end
    
    API --> ROUTER
    ROUTER -->|chat| CHAT
    ROUTER -->|jira| JIRA
    ROUTER -->|github| GITHUB
    ROUTER -->|rag| RAG
    
    CHAT --> OPENAI
    JIRA --> JIRA_API
    GITHUB --> GITHUB_API
    RAG --> MEMORY
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

The `FinanceAgent` acts as a central router that classifies user intent and dispatches requests to specialized sub-agents.

```python
# app/agent/finance_agent.py
from typing import Optional, List, Dict
from app.agents.chat_agent import ChatAgent
from app.agents.jira_agent import JiraAgent
from app.agents.github_agent import GitHubAgent

class FinanceAgent:
    """
    Central router for the Finance Chat System.
    """
    
    def __init__(self):
        # Initialize sub-agents
        self.chat_agent = ChatAgent()
        self.jira_agent = JiraAgent()
        self.github_agent = GitHubAgent()
        self.rag_agent = None  # Placeholder for RAG
        
    def _detect_agent_type(self, message: str) -> str:
        """
        Classify user intent to determine the appropriate agent.
        """
        message_lower = message.lower()
        
        # Keyword-based routing (can be upgraded to LLM-based)
        if any(k in message_lower for k in ["github", "pr", "repo"]):
            return "github"
        elif any(k in message_lower for k in ["jira", "ticket", "sprint"]):
            return "jira"
        elif any(k in message_lower for k in ["search", "document"]):
            return "rag"
        
        return "chat"

    async def run(self, message: str, history: List[Dict], thread_id: Optional[str] = None) -> str:
        """
        Process the message by routing to the correct sub-agent.
        """
        agent_type = self._detect_agent_type(message)
        
        if agent_type == "github":
            return await self.github_agent.process_query(message)
        elif agent_type == "jira":
            return await self.jira_agent.process_query(message)
        elif agent_type == "rag":
             return "RAG functionality not yet implemented"
        else:
            # Default to ChatAgent for general conversation
            return await self.chat_agent.process_query(message)
```

### 3.3 Chat Agent Design

The `ChatAgent` handles general conversational queries, financial concepts, and "chit-chat". It maintains conversation history and provides helpful, context-aware responses.

```python
# app/agents/chat_agent.py
from langchain_openai import ChatOpenAI

class ChatAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
    async def process_query(self, message: str) -> str:
        """
        Process general chat messages.
        """
        # Logic to handle chat, potentially using LangGraph for state
        response = await self.llm.ainvoke(message)
        return response.content
```

### 3.4 Dynamic LLM Management

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
    
    async def invoke(self, messages, **invoke_kwargs):
        """
        Invoke the LLM with automatic API key management
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                api_key = self._get_valid_api_key()
                
                client = openai.AsyncOpenAI(api_key=api_key)
                
                response = await client.chat.completions.create(
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

### 3.4 Background Task Integration

The system uses FastAPI's built-in `BackgroundTasks` for asynchronous processing, eliminating the need for a separate message broker like RabbitMQ or Redis. This adheres to the "Simplicity First" principle.

```python
# app/services/message_service.py
from fastapi import BackgroundTasks
from app.core.database import SessionLocal
from app.agents.finance_agent import FinanceAgent
from app.models.conversation import Message, Conversation
import json

async def process_message_background(message_id: str, user_id: str):
    """
    Background task to process message without Celery.
    Manages its own DB session.
    """
    db = SessionLocal()
    try:
        # Retrieve message and conversation
        user_msg = db.query(Message).filter(Message.id == message_id).first()
        if not user_msg:
            return

        conv = db.query(Conversation).filter(
            Conversation.id == user_msg.conversation_id, 
            Conversation.user_id == user_id
        ).first()
        
        if not conv:
            return

        # Initialize and run agent
        agent = FinanceAgent()
        reply = await agent.run(user_msg.content, [], thread_id=str(conv.id))
        
        # Save Response
        ai_msg = Message(
            conversation_id=conv.id, 
            role="assistant", 
            content=reply, 
            meta=json.dumps({"parent_message_id": message_id})
        )
        db.add(ai_msg)
        db.commit()
        
        # Update user message status
        meta = json.loads(user_msg.meta) if user_msg.meta else {}
        meta.update({"status": "completed", "response_message_id": ai_msg.id})
        user_msg.meta = json.dumps(meta)
        db.commit()
            
    except Exception as e:
        db.rollback()
        # Handle error state updates here
    finally:
        db.close()
```

### 3.5 Prompt Management System

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
      - DATABASE_URL=postgresql://user:pass@postgres:5432/finance_ai
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=finance_ai
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## 8. Security Considerations

### 8.1 API Key Management
- **Rotation**: Automatic API key refresh based on expiration
- **Isolation**: User-specific API keys with rate limiting
- **Encryption**: Keys encrypted at rest and in transit

### 8.2 Data Protection
- **PII Handling**: LLM responses sanitized for sensitive information
- **Audit Logs**: All actions logged for compliance

### 8.3 Rate Limiting
```python
# app/middleware/rate_limit.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour", "10/minute"]
)
```

## 9. Monitoring and Observability

### 9.1 Metrics Collection
```python
# app/monitoring/metrics.py
from prometheus_client import Counter, Histogram

# Agent metrics
AGENT_REQUESTS = Counter('agent_requests_total', 'Total agent requests')
AGENT_REQUEST_DURATION = Histogram('agent_request_duration_seconds', 'Request duration')
AGENT_ERRORS = Counter('agent_errors_total', 'Agent errors by type', ['error_type'])
LLM_TOKEN_USAGE = Counter('llm_tokens_total', 'LLM token usage', ['model', 'type'])
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
agent_logger = structlog.get_logger("finance_agent")
agent_logger = agent_logger.bind(
    component="agent",
    version="1.0.0"
)
```

## 10. Scaling Considerations

### 10.1 Horizontal Scaling
- **Stateless API Layer**: API servers can be scaled independently
- **PostgreSQL Read Replicas**: For checkpoint and state storage

## 11. Conclusion

The Finance Agent system provides a robust solution for intelligent financial assistance. Key strengths include:

1. **Flexible Architecture**: Modular design with clear separation of concerns
2. **Simplicity**: Direct synchronous communication for low latency
3. **Maintainability**: Externalized prompts and configuration
4. **Observability**: Comprehensive monitoring and logging
5. **Security**: Proper API key management and data protection

The system effectively leverages GPT-4 to provide accurate and helpful financial information to users.

## 12. Appendices

### 12.1 Sample Prompt Templates

**finance_advisor.md**
```markdown
---
name: finance_advisor
version: 1.0
variables:
  - user_message
  - history
---

## System:
You are a helpful and knowledgeable financial assistant. 
Your goal is to explain financial concepts clearly and provide data-driven insights.

## User:
Message: {{ user_message }}

Previous conversation:
{% for msg in history %}
{{ msg.role }}: {{ msg.content }}
{% endfor %}

## Instructions:
1. Be concise and clear.
2. If the user asks for advice, provide general principles and a disclaimer.
3. Use data when available.
```

### 12.2 Error Handling Strategy

```python
# app/errors/handlers.py
class AgentError(Exception):
    """Base exception for agent errors"""
    pass

class LLMError(AgentError):
    """LLM-related errors"""
    pass
```