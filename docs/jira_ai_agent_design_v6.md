# Jira AI Agent System Design Document

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [AI Agent Design Patterns](#ai-agent-design-patterns)
5. [API Design](#api-design)
6. [Implementation Guide](#implementation-guide)
7. [Use Cases & Examples](#use-cases--examples)
8. [Deployment & Operations](#deployment--operations)

---

## 1. Executive Summary

### 1.1 Overview
The Jira AI Agent is an intelligent backend service powered by OpenAI's GPT-4.1 that provides natural language interaction with Jira. It enables users to analyze requirements, create tickets, and assess existing tickets through conversational AI with human-in-the-loop capabilities.

### 1.2 Key Capabilities
- **Asynchronous Processing**: Non-blocking chat requests with result polling
- **Intelligent Intent Classification**: Automatic routing based on user intent
- **Jira Integration**: Full CRUD operations on Jira tickets
- **Conversation Memory**: Stateful conversations with PostgreSQL persistence
- **Human-in-the-Loop**: Interactive confirmation workflows
- **Dynamic API Management**: Automatic OpenAI API key rotation
- **Structured Outputs**: JSON and Markdown response formats

### 1.3 Technology Stack
- **AI Framework**: LangChain + LangGraph
- **LLM**: OpenAI GPT-4.1
- **Task Queue**: Celery + Redis
- **Web Framework**: FastAPI
- **Database**: PostgreSQL
- **Checkpointing**: LangGraph PostgresSaver

---

## 2. System Architecture

### 2.1 High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Client[Client Applications]
    end
    
    subgraph "API Gateway Layer"
        FastAPI[FastAPI Server]
        Auth[Authentication]
        RateLimit[Rate Limiting]
    end
    
    subgraph "Orchestration Layer"
        Celery[Celery Workers]
        Redis[(Redis Queue)]
        TaskMgr[Task Manager]
    end
    
    subgraph "AI Agent Layer"
        LangGraph[LangGraph Workflow]
        Classifier[Intent Classifier]
        ReqAnalyzer[Requirement Analyzer]
        TicketCreator[Ticket Creator]
        TicketAssessor[Ticket Assessor]
        HumanLoop[Human-in-Loop Handler]
    end
    
    subgraph "Integration Layer"
        OpenAI[OpenAI GPT-4.1]
        KeyMgr[API Key Manager]
        JiraAPI[Jira REST API]
        PromptStore[Prompt Storage]
    end
    
    subgraph "Data Layer"
        PostgreSQL[(PostgreSQL)]
        Checkpointer[PostgresSaver]
        ConvHistory[Conversation History]
        Prompts[Prompt Templates]
    end
    
    Client --> FastAPI
    FastAPI --> Auth
    Auth --> RateLimit
    RateLimit --> Redis
    FastAPI --> Celery
    Celery --> LangGraph
    
    LangGraph --> Classifier
    Classifier --> ReqAnalyzer
    Classifier --> TicketCreator
    Classifier --> TicketAssessor
    
    ReqAnalyzer --> OpenAI
    TicketCreator --> OpenAI
    TicketCreator --> HumanLoop
    TicketAssessor --> OpenAI
    TicketAssessor --> JiraAPI
    
    OpenAI --> KeyMgr
    TicketCreator --> JiraAPI
    
    LangGraph --> Checkpointer
    Checkpointer --> PostgreSQL
    PromptStore --> Prompts
    TaskMgr --> ConvHistory
    ConvHistory --> PostgreSQL
```

### 2.2 Data Flow Architecture

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant Redis as Redis Queue
    participant Celery as Celery Worker
    participant Graph as LangGraph Agent
    participant LLM as OpenAI GPT-4.1
    participant Jira as Jira API
    participant DB as PostgreSQL
    
    C->>API: POST /chat/send
    API->>Redis: Enqueue Task
    API-->>C: 202 {task_id}
    
    Redis->>Celery: Dequeue Task
    Celery->>Graph: Execute Workflow
    Graph->>DB: Load Checkpoint
    
    Graph->>Graph: Classify Intent
    Graph->>LLM: Analyze Request
    LLM-->>Graph: Classification Result
    
    alt Create Ticket
        Graph->>LLM: Extract Ticket Details
        LLM-->>Graph: Structured Data
        Graph->>Graph: Interrupt (HITL)
        Graph->>DB: Save Checkpoint
        
        C->>API: GET /chat/result/{task_id}
        API-->>C: {status: "waiting_confirmation"}
        
        C->>API: POST /chat/resume/{task_id}
        API->>Celery: Resume Task
        Celery->>Graph: Continue Workflow
        Graph->>DB: Load Checkpoint
        Graph->>Jira: Create Ticket
        Jira-->>Graph: Ticket Created
    else Assess Ticket
        Graph->>Jira: Fetch Ticket
        Jira-->>Graph: Ticket Data
        Graph->>LLM: Analyze Ticket
        LLM-->>Graph: Assessment
    end
    
    Graph->>DB: Save Final State
    Graph-->>Celery: Complete
    
    C->>API: GET /chat/result/{task_id}
    API->>DB: Query Result
    API-->>C: 200 {result}
```

---

## 3. Core Components

### 3.1 FastAPI Service

**Purpose**: RESTful API gateway handling HTTP requests and task orchestration.

```python
# api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
from datetime import datetime

app = FastAPI(title="Jira AI Agent API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: str
    
class ResumeRequest(BaseModel):
    user_input: Optional[Dict[str, Any]] = None
    confirmation: Optional[bool] = None

class ChatResponse(BaseModel):
    task_id: str
    conversation_id: str
    status: str
    created_at: datetime

@app.post("/chat/send", response_model=ChatResponse)
async def send_chat_message(request: ChatRequest):
    """
    Send a chat message for asynchronous processing.
    Returns a task_id for polling results.
    """
    task_id = str(uuid.uuid4())
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Enqueue Celery task
    from tasks.agent_tasks import process_chat_message
    task = process_chat_message.delay(
        task_id=task_id,
        conversation_id=conversation_id,
        user_id=request.user_id,
        message=request.message
    )
    
    return ChatResponse(
        task_id=task_id,
        conversation_id=conversation_id,
        status="processing",
        created_at=datetime.utcnow()
    )

@app.get("/chat/result/{task_id}")
async def get_chat_result(task_id: str):
    """
    Poll for the result of a chat task.
    Returns status and result when complete.
    """
    from tasks.agent_tasks import process_chat_message
    from celery.result import AsyncResult
    
    task_result = AsyncResult(task_id, app=process_chat_message.app)
    
    if task_result.ready():
        if task_result.successful():
            result = task_result.get()
            return JSONResponse(content={
                "status": "completed",
                "result": result
            })
        else:
            return JSONResponse(content={
                "status": "failed",
                "error": str(task_result.info)
            }, status_code=500)
    
    # Check if waiting for human input
    state = get_task_state(task_id)
    if state and state.get("interrupted"):
        return JSONResponse(content={
            "status": "waiting_confirmation",
            "message": state.get("interrupt_message"),
            "data": state.get("interrupt_data")
        })
    
    return JSONResponse(content={"status": "processing"})

@app.post("/chat/resume/{task_id}")
async def resume_chat_task(task_id: str, request: ResumeRequest):
    """
    Resume a paused workflow with user confirmation or input.
    """
    from tasks.agent_tasks import resume_chat_workflow
    
    # Resume the workflow from checkpoint
    task = resume_chat_workflow.delay(
        task_id=task_id,
        user_input=request.user_input,
        confirmation=request.confirmation
    )
    
    return JSONResponse(content={
        "status": "resumed",
        "task_id": task_id
    })

def get_task_state(task_id: str) -> Optional[Dict]:
    """Query PostgreSQL for task state."""
    from core.database import get_db
    db = next(get_db())
    # Implementation to fetch state
    return None
```

### 3.2 Celery Task Queue

**Purpose**: Asynchronous task execution with retry and monitoring capabilities.

```python
# tasks/celery_app.py
from celery import Celery
import os

celery_app = Celery(
    'jira_ai_agent',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes
    task_soft_time_limit=540,  # 9 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100
)
```

```python
# tasks/agent_tasks.py
from tasks.celery_app import celery_app
from core.agent import JiraAIAgent
from core.database import get_db
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3)
def process_chat_message(self, task_id: str, conversation_id: str, 
                         user_id: str, message: str) -> Dict[str, Any]:
    """
    Process a chat message through the LangGraph agent.
    """
    try:
        agent = JiraAIAgent()
        
        result = agent.run(
            task_id=task_id,
            conversation_id=conversation_id,
            user_id=user_id,
            message=message
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        self.retry(exc=e, countdown=60)

@celery_app.task(bind=True)
def resume_chat_workflow(self, task_id: str, user_input: Dict = None, 
                         confirmation: bool = None) -> Dict[str, Any]:
    """
    Resume a paused workflow from checkpoint.
    """
    try:
        agent = JiraAIAgent()
        
        result = agent.resume(
            task_id=task_id,
            user_input=user_input,
            confirmation=confirmation
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Resume task {task_id} failed: {str(e)}")
        raise
```

### 3.3 Dynamic OpenAI Client with API Key Management

**Purpose**: Automatic API key rotation and expiration handling.

```python
# core/openai_client.py
from langchain_openai import ChatOpenAI
from typing import Optional
from datetime import datetime, timedelta
import logging
import asyncio

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages OpenAI API keys with automatic refresh."""
    
    def __init__(self):
        self.current_key: Optional[str] = None
        self.key_expiry: Optional[datetime] = None
        self.refresh_lock = asyncio.Lock()
    
    def get_api_key(self) -> str:
        """Get current API key, refreshing if expired."""
        if self.is_key_expired():
            self.refresh_api_key()
        return self.current_key
    
    def is_key_expired(self) -> bool:
        """Check if the current API key has expired."""
        if not self.current_key or not self.key_expiry:
            return True
        return datetime.utcnow() >= self.key_expiry
    
    def refresh_api_key(self):
        """Fetch a new API key from secure storage."""
        # In production, fetch from secrets manager (AWS Secrets Manager, Vault, etc.)
        from core.config import get_openai_api_key
        
        logger.info("Refreshing OpenAI API key")
        self.current_key = get_openai_api_key()
        # Assume keys expire after 24 hours
        self.key_expiry = datetime.utcnow() + timedelta(hours=24)
        logger.info(f"New API key set, expires at {self.key_expiry}")

class DynamicChatOpenAI(ChatOpenAI):
    """ChatOpenAI wrapper with dynamic API key management."""
    
    def __init__(self, key_manager: APIKeyManager, **kwargs):
        self.key_manager = key_manager
        # Initialize with current key
        api_key = self.key_manager.get_api_key()
        super().__init__(openai_api_key=api_key, **kwargs)
    
    def _get_ls_params(self, *args, **kwargs):
        """Override to refresh API key before each call."""
        # Refresh key if expired
        new_key = self.key_manager.get_api_key()
        if new_key != self.openai_api_key:
            logger.info("Updating OpenAI API key in client")
            self.openai_api_key = new_key
        
        return super()._get_ls_params(*args, **kwargs)

# Global key manager instance
_key_manager = APIKeyManager()

def get_llm(model: str = "gpt-4-turbo-preview", temperature: float = 0.7) -> DynamicChatOpenAI:
    """Factory function to create LLM instance with key management."""
    return DynamicChatOpenAI(
        key_manager=_key_manager,
        model=model,
        temperature=temperature,
        request_timeout=120
    )
```

### 3.4 Prompt Management System

**Purpose**: Centralized prompt storage and version control.

```python
# core/prompt_manager.py
from typing import Dict, Optional
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompt templates stored as markdown files."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.cache: Dict[str, str] = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """Load all prompt templates from disk."""
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory {self.prompts_dir} not found")
            return
        
        for prompt_file in self.prompts_dir.glob("*.md"):
            key = prompt_file.stem
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.cache[key] = f.read()
            logger.info(f"Loaded prompt: {key}")
    
    def get_prompt(self, key: str, **kwargs) -> str:
        """
        Get a prompt template and format with variables.
        
        Args:
            key: Prompt identifier
            **kwargs: Variables to format into the prompt
        
        Returns:
            Formatted prompt string
        """
        if key not in self.cache:
            # Try to load from database as fallback
            prompt = self._load_from_db(key)
            if prompt:
                self.cache[key] = prompt
            else:
                raise ValueError(f"Prompt '{key}' not found")
        
        template = self.cache[key]
        
        # Simple variable substitution
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable in prompt {key}: {e}")
            raise
    
    def _load_from_db(self, key: str) -> Optional[str]:
        """Load prompt from database/config center."""
        # Implement database lookup
        # For example, using SQLAlchemy:
        # from core.database import SessionLocal, PromptTemplate
        # session = SessionLocal()
        # result = session.query(PromptTemplate).filter_by(key=key).first()
        # return result.content if result else None
        return None
    
    def reload(self):
        """Reload all prompts from disk."""
        self.cache.clear()
        self._load_prompts()

# Global prompt manager
_prompt_manager = PromptManager()

def get_prompt(key: str, **kwargs) -> str:
    """Convenience function to get a prompt."""
    return _prompt_manager.get_prompt(key, **kwargs)
```

**Example Prompt Templates:**

```markdown
<!-- prompts/intent_classifier.md -->
# Intent Classification

You are an intent classifier for a Jira AI Agent. Analyze the user's message and classify it into one of the following categories:

## Intent Categories:
1. **requirement_analysis**: User wants to analyze requirements or get structured output
2. **create_ticket**: User wants to create a new Jira ticket
3. **assess_ticket**: User wants to assess or review an existing ticket
4. **general_query**: General questions or chitchat

## Instructions:
- Respond with ONLY a JSON object
- Include: intent, confidence (0-1), reasoning

## User Message:
{message}

## Response Format:
```json
{{
  "intent": "<intent_category>",
  "confidence": 0.95,
  "reasoning": "Brief explanation"
}}
```
```

```markdown
<!-- prompts/requirement_analyzer.md -->
# Requirement Analysis

You are a senior business analyst specializing in software requirements. Analyze the provided requirement and produce a structured output.

## User Requirement:
{requirement}

## Analysis Framework:
1. **Functional Requirements**: Core features and capabilities
2. **Non-Functional Requirements**: Performance, security, scalability
3. **Acceptance Criteria**: Measurable success criteria
4. **Dependencies**: External systems or prerequisites
5. **Risks**: Potential challenges or blockers

## Output Format:
{output_format}

Provide a comprehensive analysis in the requested format.
```

```markdown
<!-- prompts/ticket_creator.md -->
# Jira Ticket Creation

You are a Jira expert. Extract ticket information from the user's request and structure it for Jira API.

## User Request:
{message}

## Conversation Context:
{context}

## Required Fields:
- **summary**: Brief title (max 255 chars)
- **description**: Detailed description with formatting
- **issue_type**: Story, Task, Bug, Epic
- **priority**: Highest, High, Medium, Low, Lowest
- **labels**: Array of relevant tags

## Optional Fields:
- **assignee**: Username or email
- **reporter**: Username or email
- **components**: Array of component names
- **fix_versions**: Array of version names

## Instructions:
1. Extract all relevant information
2. Use Jira markdown formatting for description
3. Infer missing fields intelligently
4. Return JSON matching Jira API schema

## Response Format:
```json
{{
  "fields": {{
    "project": {{"key": "PROJ"}},
    "summary": "...",
    "description": "...",
    "issuetype": {{"name": "Story"}},
    "priority": {{"name": "High"}},
    "labels": ["backend", "api"]
  }}
}}
```
```

```markdown
<!-- prompts/ticket_assessor.md -->
# Jira Ticket Assessment

You are a technical project manager assessing Jira tickets for quality and completeness.

## Ticket Information:
{ticket_data}

## Assessment Criteria:
1. **Clarity**: Is the ticket clear and unambiguous?
2. **Completeness**: Are all necessary details included?
3. **Acceptance Criteria**: Are success criteria well-defined?
4. **Estimability**: Can developers estimate effort?
5. **Dependencies**: Are dependencies identified?
6. **Risk Level**: What are the risks?

## Analysis Depth:
- Identify gaps and missing information
- Suggest improvements
- Rate overall ticket quality (1-10)
- Provide actionable recommendations

## Output Format:
{output_format}

Provide a thorough assessment with specific, actionable feedback.
```

### 3.5 LangGraph Agent Implementation

**Purpose**: Core AI agent with workflow orchestration and checkpointing.

```python
# core/agent.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence, Dict, Any, Optional
import operator
import logging
from core.openai_client import get_llm
from core.prompt_manager import get_prompt
from core.jira_client import JiraClient
from core.database import get_db_connection_string

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State object for the agent workflow."""
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    intent: Optional[str]
    confidence: Optional[float]
    ticket_data: Optional[Dict[str, Any]]
    jira_response: Optional[Dict[str, Any]]
    user_confirmation: Optional[bool]
    final_output: Optional[Dict[str, Any]]
    error: Optional[str]

class JiraAIAgent:
    """Main agent orchestrating the workflow."""
    
    def __init__(self):
        self.llm = get_llm(model="gpt-4-turbo-preview")
        self.jira_client = JiraClient()
        self.checkpointer = PostgresSaver.from_conn_string(
            get_db_connection_string()
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("analyze_requirement", self.analyze_requirement)
        workflow.add_node("prepare_ticket", self.prepare_ticket)
        workflow.add_node("confirm_ticket", self.confirm_ticket)
        workflow.add_node("create_ticket", self.create_ticket)
        workflow.add_node("assess_ticket", self.assess_ticket)
        workflow.add_node("handle_error", self.handle_error)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_intent,
            {
                "requirement_analysis": "analyze_requirement",
                "create_ticket": "prepare_ticket",
                "assess_ticket": "assess_ticket",
                "general_query": END,
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("analyze_requirement", END)
        workflow.add_edge("prepare_ticket", "confirm_ticket")
        
        # Interrupt before creating ticket (Human-in-the-Loop)
        workflow.add_edge("confirm_ticket", "create_ticket")
        workflow.add_edge("create_ticket", END)
        workflow.add_edge("assess_ticket", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["create_ticket"]  # HITL checkpoint
        )
    
    def classify_intent(self, state: AgentState) -> AgentState:
        """Classify user intent using LLM."""
        logger.info("Classifying user intent")
        
        last_message = state["messages"][-1].content
        prompt = get_prompt("intent_classifier", message=last_message)
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            import json
            result = json.loads(response.content)
            
            state["intent"] = result["intent"]
            state["confidence"] = result["confidence"]
            
            logger.info(f"Intent: {result['intent']}, Confidence: {result['confidence']}")
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            state["intent"] = "error"
            state["error"] = str(e)
        
        return state
    
    def route_intent(self, state: AgentState) -> str:
        """Route based on classified intent."""
        intent = state.get("intent", "error")
        confidence = state.get("confidence", 0)
        
        if confidence < 0.6:
            logger.warning(f"Low confidence ({confidence}) for intent: {intent}")
        
        return intent
    
    def analyze_requirement(self, state: AgentState) -> AgentState:
        """Analyze requirements and provide structured output."""
        logger.info("Analyzing requirements")
        
        last_message = state["messages"][-1].content
        
        # User can specify output format
        output_format = "markdown"  # or "json"
        
        prompt = get_prompt(
            "requirement_analyzer",
            requirement=last_message,
            output_format=output_format
        )
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        state["final_output"] = {
            "type": "requirement_analysis",
            "content": response.content,
            "format": output_format
        }
        
        return state
    
    def prepare_ticket(self, state: AgentState) -> AgentState:
        """Prepare Jira ticket data from user message."""
        logger.info("Preparing ticket data")
        
        last_message = state["messages"][-1].content
        context = self._get_conversation_context(state["messages"])
        
        prompt = get_prompt(
            "ticket_creator",
            message=last_message,
            context=context
        )
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            import json
            ticket_data = json.loads(response.content)
            
            state["ticket_data"] = ticket_data
            logger.info(f"Ticket data prepared: {ticket_data.get('fields', {}).get('summary')}")
            
        except Exception as e:
            logger.error(f"Ticket preparation failed: {e}")
            state["error"] = str(e)
        
        return state
    
    def confirm_ticket(self, state: AgentState) -> AgentState:
        """Prepare confirmation message for user."""
        logger.info("Awaiting user confirmation for ticket creation")
        
        ticket_data = state.get("ticket_data", {})
        fields = ticket_data.get("fields", {})
        
        # This will trigger an interrupt
        state["final_output"] = {
            "type": "confirmation_required",
            "message": "Please review the ticket details before creation:",
            "ticket_summary": fields.get("summary"),
            "ticket_description": fields.get("description"),
            "issue_type": fields.get("issuetype", {}).get("name"),
            "priority": fields.get("priority", {}).get("name"),
            "labels": fields.get("labels", [])
        }
        
        return state
    
    def create_ticket(self, state: AgentState) -> AgentState:
        """Create ticket in Jira after confirmation."""
        logger.info("Creating Jira ticket")
        
        # Check user confirmation
        if not state.get("user_confirmation"):
            state["final_output"] = {
                "type": "cancelled",
                "message": "Ticket creation cancelled by user"
            }
            return state
        
        ticket_data = state.get("ticket_data", {})
        
        try:
            jira_response = self.jira_client.create_issue(ticket_data)
            
            state["jira_response"] = jira_response
            state["final_output"] = {
                "type": "ticket_created",
                "ticket_key": jira_response.get("key"),
                "ticket_url": jira_response.get("self"),
                "message": f"Successfully created ticket {jira_response.get('key')}"
            }
            
            logger.info(f"Ticket created: {jira_response.get('key')}")
            
        except Exception as e:
            logger.error(f"Ticket creation failed: {e}")
            state["error"] = str(e)
            state["final_output"] = {
                "type": "error",
                "message": f"Failed to create ticket: {str(e)}"
            }
        
        return state
    
    def assess_ticket(self, state: AgentState) -> AgentState:
        """Assess an existing Jira ticket."""
        logger.info("Assessing Jira ticket")
        
        last_message = state["messages"][-1].content
        
        # Extract ticket number from message
        import re
        ticket_match = re.search(r'([A-Z]+-\d+)', last_message)
        
        if not ticket_match:
            state["error"] = "No valid Jira ticket number found"
            return state
        
        ticket_key = ticket_match.group(1)
        
        try:
            # Fetch ticket from Jira
            ticket_info = self.jira_client.get_issue(ticket_key)
            
            # Prepare ticket data for LLM
            ticket_data = self._format_ticket_for_assessment(ticket_info)
            
            prompt = get_prompt(
                "ticket_assessor",
                ticket_data=ticket_data,
                output_format="markdown"
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            state["final_output"] = {
                "type": "ticket_assessment",
                "ticket_key": ticket_key,
                "assessment": response.content
            }
            
            logger.info(f"Ticket {ticket_key} assessed successfully")
            
        except Exception as e:
            logger.error(f"Ticket assessment failed: {e}")
            state["error"] = str(e)
        
        return state
    
    def handle_error(self, state: AgentState) -> AgentState:
        """Handle errors gracefully."""
        error = state.get("error", "Unknown error")
        logger.error(f"Handling error: {error}")
        
        state["final_output"] = {
            "type": "error",
            "message": f"An error occurred: {error}"
        }
        
        return state
    
    def _get_conversation_context(self, messages: Sequence) -> str:
        """Extract conversation context for better understanding."""
        context_messages = messages[-5:]  # Last 5 messages
        context = []
        
        for msg in context_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            context.append(f"{role}: {msg.content}")
        
        return "\n".join(context)
    
    def _format_ticket_for_assessment(self, ticket_info: Dict) -> str:
        """Format Jira ticket data for LLM analysis."""
        fields = ticket_info.get("fields", {})
        
        formatted = f"""
**Ticket Key**: {ticket_info.get('key')}
**Summary**: {fields.get('summary')}
**Status**: {fields.get('status', {}).get('name')}
**Issue Type**: {fields.get('issuetype', {}).get('name')}
**Priority**: {fields.get('priority', {}).get('name')}
**Description**:
{fields.get('description', 'No description')}

**Reporter**: {fields.get('reporter', {}).get('displayName')}
**Assignee**: {fields.get('assignee', {}).get('displayName', 'Unassigned')}
**Labels**: {', '.join(fields.get('labels', []))}
**Components**: {', '.join([c.get('name') for c in fields.get('components', [])])}
"""
        return formatted
    
    def run(self, task_id: str, conversation_id: str, 
            user_id: str, message: str) -> Dict[str, Any]:
        """Execute the agent workflow."""
        config = {
            "configurable": {
                "thread_id": conversation_id,
                "checkpoint_ns": task_id
            }
        }
        
        initial_state = {
            "messages": [HumanMessage(content=message)]
        }
        
        try:
            result = self.graph.invoke(initial_state, config=config)
            return result.get("final_output", {})
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "type": "error",
                "message": str(e)
            }
    
    def resume(self, task_id: str, user_input: Dict = None, 
               confirmation: bool = None) -> Dict[str, Any]:
        """Resume workflow from checkpoint."""
        # Get the checkpoint
        config = {
            "configurable": {
                "checkpoint_ns": task_id
            }
        }
        
        # Update state with user input
        state_update = {}
        if confirmation is not None:
            state_update["user_confirmation"] = confirmation
        if user_input:
            state_update.update(user_input)
        
        try:
            # Continue from interrupted point
            result = self.graph.invoke(state_update, config=config)
            return result.get("final_output", {})
            
        except Exception as e:
            logger.error(f"Resume execution failed: {e}")
            return {
                "type": "error",
                "message": str(e)
            }
    
    def visualize(self) -> str:
        """Generate Mermaid diagram of the workflow."""
        return self.graph.get_graph().draw_mermaid()
```

### 3.6 Jira API Client

**Purpose**: Abstraction layer for Jira REST API interactions.

```python
# core/jira_client.py
import requests
from typing import Dict, Any, Optional, List
import logging
from core.config import get_jira_config

logger = logging.getLogger(__name__)

class JiraClient:
    """Client for Jira REST API interactions."""
    
    def __init__(self):
        config = get_jira_config()
        self.base_url = config["base_url"]
        self.email = config["email"]
        self.api_token = config["api_token"]
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create authenticated session."""
        session = requests.Session()
        session.auth = (self.email, self.api_token)
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        return session
    
    def create_issue(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new Jira issue.
        
        Args:
            issue_data: Issue fields matching Jira API schema
        
        Returns:
            Created issue data with key and ID
        """
        url = f"{self.base_url}/rest/api/3/issue"
        
        try:
            response = self.session.post(url, json=issue_data)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Created Jira issue: {result['key']}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create Jira issue: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def get_issue(self, issue_key: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fetch an existing Jira issue.
        
        Args:
            issue_key: Jira issue key (e.g., PROJ-123)
            fields: Optional list of fields to return
        
        Returns:
            Issue data
        """
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        
        params = {}
        if fields:
            params["fields"] = ",".join(fields)
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Retrieved Jira issue: {issue_key}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve Jira issue {issue_key}: {e}")
            raise
    
    def update_issue(self, issue_key: str, update_data: Dict[str, Any]) -> bool:
        """Update an existing Jira issue."""
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        
        try:
            response = self.session.put(url, json=update_data)
            response.raise_for_status()
            logger.info(f"Updated Jira issue: {issue_key}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update Jira issue {issue_key}: {e}")
            raise
    
    def add_comment(self, issue_key: str, comment: str) -> Dict[str, Any]:
        """Add a comment to a Jira issue."""
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}/comment"
        
        payload = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": comment
                            }
                        ]
                    }
                ]
            }
        }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to add comment to {issue_key}: {e}")
            raise
```

### 3.7 Database Schema and Models

**Purpose**: PostgreSQL schema for checkpointing and conversation history.

```python
# core/database.py
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class ConversationHistory(Base):
    """Store conversation history."""
    __tablename__ = "conversation_history"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, index=True)
    user_id = Column(String, index=True)
    task_id = Column(String, unique=True, index=True)
    message = Column(Text)
    response = Column(JSON)
    status = Column(String)  # processing, completed, failed, waiting_confirmation
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PromptTemplate(Base):
    """Store prompt templates."""
    __tablename__ = "prompt_templates"
    
    id = Column(String, primary_key=True)
    key = Column(String, unique=True, index=True)
    content = Column(Text)
    version = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class TaskState(Base):
    """Store task execution state."""
    __tablename__ = "task_states"
    
    task_id = Column(String, primary_key=True)
    conversation_id = Column(String, index=True)
    state_data = Column(JSON)
    interrupted = Column(Boolean, default=False)
    interrupt_message = Column(Text)
    interrupt_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/jira_ai_agent")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Database session generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_connection_string() -> str:
    """Get database connection string for checkpointer."""
    return DATABASE_URL

def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
```

---

## 4. AI Agent Design Patterns

### 4.1 Planning Pattern

**Purpose**: Break down complex tasks into subtasks.

```python
# patterns/planning.py
from typing import List, Dict
from langchain_core.messages import HumanMessage
from core.openai_client import get_llm

class PlanningAgent:
    """Agent that creates execution plans for complex tasks."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.3)
    
    def create_plan(self, task_description: str) -> List[Dict[str, str]]:
        """
        Create a step-by-step plan for a complex task.
        
        Returns:
            List of steps with descriptions and dependencies
        """
        prompt = f"""
You are a task planning expert. Break down the following task into concrete steps:

Task: {task_description}

Provide a plan as a JSON array of steps, where each step has:
- step_number: integer
- description: what to do
- dependencies: list of step numbers that must complete first
- estimated_complexity: low/medium/high

Example format:
[
  {{"step_number": 1, "description": "...", "dependencies": [], "estimated_complexity": "low"}},
  {{"step_number": 2, "description": "...", "dependencies": [1], "estimated_complexity": "medium"}}
]
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        import json
        plan = json.loads(response.content)
        return plan
```

### 4.2 ReAct Pattern (Reasoning + Acting)

**Purpose**: Iterative reasoning and action execution.

```python
# patterns/react.py
from typing import Optional, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from core.openai_client import get_llm

class ReActAgent:
    """Agent using Reasoning + Acting pattern."""
    
    def __init__(self):
        self.llm = get_llm()
        self.max_iterations = 5
    
    def run(self, task: str, tools: Dict[str, callable]) -> str:
        """
        Execute task using ReAct loop.
        
        Args:
            task: Task description
            tools: Dictionary of available tools/functions
        
        Returns:
            Final answer
        """
        system_prompt = """
You are an AI agent using the ReAct (Reasoning + Acting) framework.

For each iteration:
1. Thought: Reason about what to do next
2. Action: Choose a tool and provide input
3. Observation: Observe the result

Continue until you have the final answer, then output:
Answer: [final answer]

Available tools:
{tool_descriptions}
"""
        
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__}" 
            for name, func in tools.items()
        ])
        
        messages = [
            SystemMessage(content=system_prompt.format(tool_descriptions=tool_descriptions)),
            HumanMessage(content=f"Task: {task}")
        ]
        
        for iteration in range(self.max_iterations):
            response = self.llm.invoke(messages)
            content = response.content
            
            # Check if we have a final answer
            if "Answer:" in content:
                answer = content.split("Answer:")[-1].strip()
                return answer
            
            # Parse action
            if "Action:" in content:
                action_text = content.split("Action:")[1].split("Observation:")[0].strip()
                # Parse action and execute tool
                # This is simplified - real implementation would parse more carefully
                for tool_name, tool_func in tools.items():
                    if tool_name in action_text:
                        result = tool_func(action_text)
                        messages.append(HumanMessage(content=f"Observation: {result}"))
                        break
        
        return "Max iterations reached without answer"
```

### 4.3 Classification Pattern

**Purpose**: Route requests to specialized handlers.

```python
# patterns/classification.py
from enum import Enum
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from core.openai_client import get_llm
import json

class IntentType(Enum):
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    CREATE_TICKET = "create_ticket"
    ASSESS_TICKET = "assess_ticket"
    GENERAL_QUERY = "general_query"

class IntentClassifier:
    """Classifies user intent with high accuracy."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        self.intent_cache = {}
    
    def classify(self, message: str, context: str = "") -> Dict[str, Any]:
        """
        Classify user intent with confidence score.
        
        Returns:
            Dict with intent, confidence, and extracted entities
        """
        prompt = f"""
Classify the user's intent. Consider the context if provided.

User message: {message}
Context: {context}

Return JSON:
{{
  "intent": "requirement_analysis|create_ticket|assess_ticket|general_query",
  "confidence": 0.0-1.0,
  "entities": {{}},  // Extract relevant entities (ticket numbers, names, etc.)
  "reasoning": "Brief explanation"
}}
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)
        
        # Add to cache for analysis
        cache_key = hash(message)
        self.intent_cache[cache_key] = result
        
        return result
```

---

## 5. API Design

### 5.1 RESTful Endpoints

```python
# Complete API specification
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

app = FastAPI(
    title="Jira AI Agent API",
    version="1.0.0",
    description="Intelligent Jira interaction powered by GPT-4.1"
)

security = HTTPBearer()

class TaskStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_CONFIRMATION = "waiting_confirmation"

class OutputFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    user_id: str
    output_format: OutputFormat = OutputFormat.MARKDOWN
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Create a ticket for implementing user authentication",
                "user_id": "user@example.com",
                "output_format": "json"
            }
        }

class ResumeRequest(BaseModel):
    user_input: Optional[Dict[str, Any]] = None
    confirmation: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "confirmation": True
            }
        }

class ChatResponse(BaseModel):
    task_id: str
    conversation_id: str
    status: TaskStatus
    created_at: datetime

class ResultResponse(BaseModel):
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

# Endpoints implementation continues...
```

### 5.2 API Usage Examples

```python
# Example: Complete workflow
import requests
import time

BASE_URL = "http://localhost:8000"
API_KEY = "your_api_key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 1. Send chat request
response = requests.post(
    f"{BASE_URL}/chat/send",
    json={
        "message": "Create a high-priority ticket for implementing OAuth2 authentication with JWT tokens",
        "user_id": "john@example.com",
        "output_format": "json"
    },
    headers=headers
)

task_data = response.json()
task_id = task_data["task_id"]
print(f"Task created: {task_id}")

# 2. Poll for result
while True:
    response = requests.get(
        f"{BASE_URL}/chat/result/{task_id}",
        headers=headers
    )
    
    result = response.json()
    status = result["status"]
    
    print(f"Status: {status}")
    
    if status == "waiting_confirmation":
        # 3. Review and confirm
        print("Ticket details:")
        print(result["data"])
        
        confirm = input("Confirm ticket creation? (y/n): ")
        
        # 4. Resume workflow
        response = requests.post(
            f"{BASE_URL}/chat/resume/{task_id}",
            json={
                "confirmation": confirm.lower() == "y"
            },
            headers=headers
        )
        
        print("Resuming workflow...")
    
    elif status == "completed":
        print("Final result:")
        print(result["result"])
        break
    
    elif status == "failed":
        print(f"Error: {result['error']}")
        break
    
    time.sleep(2)
```

---

## 6. Implementation Guide

### 6.1 Workflow Diagram

The complete LangGraph workflow can be visualized as follows:

```mermaid
graph TD
    Start([User Request]) --> ClassifyIntent[Classify Intent]
    
    ClassifyIntent --> RouteDecision{Route by Intent}
    
    RouteDecision -->|requirement_analysis| AnalyzeReq[Analyze Requirement]
    RouteDecision -->|create_ticket| PrepareTicket[Prepare Ticket Data]
    RouteDecision -->|assess_ticket| AssessTicket[Assess Ticket]
    RouteDecision -->|general_query| GeneralResp[General Response]
    RouteDecision -->|error| HandleError[Handle Error]
    
    PrepareTicket --> ConfirmTicket[Prepare Confirmation]
    ConfirmTicket --> Interrupt{{Human-in-the-Loop<br/>Checkpoint}}
    
    Interrupt -->|User Reviews| WaitConfirm[Wait for Confirmation]
    WaitConfirm -->|Approved| CreateTicket[Create in Jira]
    WaitConfirm -->|Rejected| Cancel[Cancel Creation]
    
    AssessTicket --> FetchJira[Fetch from Jira]
    FetchJira --> AnalyzeTicket[LLM Analysis]
    
    AnalyzeReq --> End([Return Result])
    GeneralResp --> End
    CreateTicket --> End
    Cancel --> End
    AnalyzeTicket --> End
    HandleError --> End
    
    style Interrupt fill:#ffcc00
    style ClassifyIntent fill:#4CAF50
    style CreateTicket fill:#2196F3
    style HandleError fill:#f44336
```

### 6.2 Project Structure

```
jira-ai-agent/
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── routes/
│   │   ├── chat.py             # Chat endpoints
│   │   └── health.py           # Health check
│   └── dependencies.py         # Dependency injection
│
├── core/
│   ├── __init__.py
│   ├── agent.py                # Main LangGraph agent
│   ├── openai_client.py        # Dynamic OpenAI client
│   ├── jira_client.py          # Jira API client
│   ├── prompt_manager.py       # Prompt management
│   ├── database.py             # Database models
│   └── config.py               # Configuration
│
├── tasks/
│   ├── __init__.py
│   ├── celery_app.py           # Celery configuration
│   └── agent_tasks.py          # Celery tasks
│
├── patterns/
│   ├── __init__.py
│   ├── planning.py             # Planning pattern
│   ├── react.py                # ReAct pattern
│   └── classification.py       # Classification pattern
│
├── prompts/
│   ├── intent_classifier.md
│   ├── requirement_analyzer.md
│   ├── ticket_creator.md
│   └── ticket_assessor.md
│
├── tests/
│   ├── test_api.py
│   ├── test_agent.py
│   └── test_jira_client.py
│
├── alembic/                    # Database migrations
│   └── versions/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

### 6.3 Configuration Management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_KEY: str
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_TEMPERATURE: float = 0.7
    
    # Jira
    JIRA_BASE_URL: str
    JIRA_EMAIL: str
    JIRA_API_TOKEN: str
    JIRA_PROJECT_KEY: str
    
    # Agent Configuration
    MAX_ITERATIONS: int = 10
    CHECKPOINT_ENABLED: bool = True
    
    # Prompts
    PROMPTS_DIR: str = "prompts"
    PROMPTS_RELOAD_INTERVAL: int = 300  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True

_settings: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def get_openai_api_key() -> str:
    """Fetch OpenAI API key from secure storage."""
    # In production, fetch from AWS Secrets Manager, Vault, etc.
    return get_settings().OPENAI_API_KEY

def get_jira_config() -> dict:
    """Get Jira configuration."""
    settings = get_settings()
    return {
        "base_url": settings.JIRA_BASE_URL,
        "email": settings.JIRA_EMAIL,
        "api_token": settings.JIRA_API_TOKEN,
        "project_key": settings.JIRA_PROJECT_KEY
    }
```

### 6.4 Environment Variables

```bash
# .env.example
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your_secret_api_key

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/jira_ai_agent

# Redis
REDIS_URL=redis://localhost:6379/0

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.7

# Jira
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your_jira_api_token
JIRA_PROJECT_KEY=PROJ

# Agent
MAX_ITERATIONS=10
CHECKPOINT_ENABLED=true
PROMPTS_DIR=prompts
```

---

## 7. Use Cases & Examples

### 7.1 Use Case 1: Requirement Analysis

**Scenario**: Product manager wants to analyze a complex feature requirement.

**Input**:
```
Analyze the following requirement: "Users should be able to export their data 
in multiple formats (PDF, CSV, JSON) with customizable filters and scheduling 
options. The export should happen asynchronously and users should receive 
email notifications when complete."
```

**Agent Workflow**:
1. Classifies as `requirement_analysis`
2. Extracts functional and non-functional requirements
3. Identifies dependencies and risks
4. Returns structured analysis

**Output** (Markdown format):
```markdown
# Requirement Analysis: Data Export Feature

## Functional Requirements
1. **Multi-format Export**
   - Support PDF, CSV, and JSON formats
   - Format selection in UI
   - Format-specific rendering engines

2. **Customizable Filters**
   - Date range selection
   - Field selection
   - Custom query builder

3. **Scheduling Options**
   - One-time exports
   - Recurring schedules (daily, weekly, monthly)
   - Timezone handling

4. **Async Processing**
   - Background job queue
   - Progress tracking
   - Cancel functionality

5. **Email Notifications**
   - Success notifications with download link
   - Failure notifications with error details
   - Configurable notification preferences

## Non-Functional Requirements
- **Performance**: Handle exports up to 100K records
- **Scalability**: Support concurrent exports
- **Security**: Secure download links with expiration
- **Reliability**: Retry mechanism for failed exports

## Technical Dependencies
- Background job system (Celery/RQ)
- Email service integration
- File storage (S3/Cloud Storage)
- Export libraries (ReportLab, Pandas)

## Acceptance Criteria
1. User can select export format from dropdown
2. User can apply at least 3 filter types
3. Export completes within 5 minutes for 10K records
4. Email sent within 1 minute of completion
5. Download link expires after 24 hours

## Risks & Challenges
- **High**: Large dataset exports may timeout
- **Medium**: Complex PDF formatting
- **Low**: Email deliverability issues

## Recommendations
1. Implement pagination for large exports
2. Use streaming for better memory management
3. Add progress indicators in UI
4. Implement rate limiting per user
```

### 7.2 Use Case 2: Create Jira Ticket with Human-in-the-Loop

**Scenario**: Developer wants to create a bug ticket with AI assistance.

**Input**:
```
Create a high-priority bug ticket: The login page shows a 500 error 
when users enter special characters in the password field. This is 
blocking production deployment.
```

**Agent Workflow**:

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Agent
    participant LLM
    participant HITL as Human-in-Loop
    participant Jira
    
    User->>API: Create bug ticket request
    API->>Agent: Process message
    Agent->>LLM: Extract ticket details
    LLM-->>Agent: Structured ticket data
    Agent->>Agent: Prepare confirmation
    Agent-->>HITL: Interrupt workflow
    HITL-->>User: Show ticket preview
    User->>HITL: Confirm creation
    HITL->>Agent: Resume with confirmation
    Agent->>Jira: Create issue
    Jira-->>Agent: Ticket KEY-123
    Agent-->>User: Success response
```

**Step 1 - Initial Response** (Status: `waiting_confirmation`):
```json
{
  "status": "waiting_confirmation",
  "message": "Please review the ticket details before creation:",
  "ticket_summary": "Login page returns 500 error with special characters in password",
  "ticket_description": "## Bug Description\n\nThe login page throws a 500 Internal Server Error when users attempt to log in with passwords containing special characters.\n\n## Impact\n- Blocking production deployment\n- Affects all users with special characters in passwords\n\n## Steps to Reproduce\n1. Navigate to login page\n2. Enter username\n3. Enter password with special characters (e.g., @, #, $)\n4. Click Login\n5. Observe 500 error\n\n## Expected Behavior\nLogin should succeed or show appropriate validation error\n\n## Actual Behavior\nServer returns 500 Internal Server Error",
  "issue_type": "Bug",
  "priority": "High",
  "labels": ["login", "authentication", "production-blocker"]
}
```

**