# Jira AI Agent: System Design Document

## 1. Overview

### 1.1 System Purpose
The Jira AI Agent is a sophisticated backend service that leverages GPT-4.1 to provide intelligent Jira ticket management capabilities. It combines natural language processing with Jira API integration to automate ticket creation, analysis, and management workflows.

### 1.2 Key Architecture Decisions
- **Event-Driven Async Architecture**: Using Celery for distributed task processing
- **Stateful Agent Workflows**: LangGraph with persistent checkpoints for long-running conversations
- **Dynamic Model Management**: Runtime API key validation and refresh
- **Human-in-the-Loop**: Interruptible workflows with resume capability
- **Prompt Management**: Externalized prompt templates for maintainability

## 2. System Architecture

```mermaid
graph TB
    subgraph "External Systems"
        JIRA[Jira Cloud API]
        OPENAI[OpenAI GPT-4.1]
        CONFIG[Config Center/DB]
    end
    
    subgraph "API Layer"
        API[FastAPI Application]
        CHAT_API[/"POST /chat<br/>Async Request"/]
        QUERY_API[/"GET /chat/{task_id}<br/>Query Result"/]
        RESUME_API[/"POST /chat/{task_id}/resume<br/>Resume Workflow"/]
    end
    
    subgraph "Processing Layer"
        CELERY[Celery Workers]
        TASK_QUEUE[Redis Task Queue]
        AGENT[Jira AI Agent]
        GRAPH[LangGraph Workflow]
    end
    
    subgraph "Data Layer"
        POSTGRES[(PostgreSQL)]
        CHECKPOINTS[Graph Checkpoints]
        PROMPTS[Prompt Templates]
        HISTORY[Conversation History]
    end
    
    API --> TASK_QUEUE
    TASK_QUEUE --> CELERY
    CELERY --> AGENT
    AGENT --> GRAPH
    GRAPH --> OPENAI
    GRAPH --> JIRA
    AGENT --> CONFIG
    POSTGRES --> CHECKPOINTS
    POSTGRES --> HISTORY
    POSTGRES --> PROMPTS
```

## 3. Core Components

### 3.1 API Layer Design

```python
# app/api/endpoints.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: str
    metadata: Optional[dict] = None

class ResumeRequest(BaseModel):
    user_input: str
    parameters: Optional[dict] = None

@router.post("/chat")
async def create_chat_session(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Async endpoint to submit chat requests
    """
    task_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    # Queue the task for processing
    background_tasks.add_task(
        process_chat_request,
        task_id=task_id,
        session_id=session_id,
        message=request.message,
        user_id=request.user_id,
        metadata=request.metadata
    )
    
    return {
        "task_id": task_id,
        "session_id": session_id,
        "status": "processing",
        "created_at": datetime.utcnow().isoformat()
    }

@router.get("/chat/{task_id}")
async def get_chat_result(task_id: str):
    """
    Query the result of an async chat request
    """
    result = await result_store.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return result

@router.post("/chat/{task_id}/resume")
async def resume_workflow(task_id: str, resume_request: ResumeRequest):
    """
    Resume a paused workflow with user input
    """
    # Validate task exists and is in paused state
    task_state = await checkpoint_store.get_state(task_id)
    if not task_state or task_state.get("status") != "paused":
        raise HTTPException(
            status_code=400,
            detail="Task is not in a resumable state"
        )
    
    # Send resume signal to workflow
    background_tasks.add_task(
        resume_agent_workflow,
        task_id=task_id,
        user_input=resume_request.user_input,
        parameters=resume_request.parameters
    )
    
    return {"status": "resumed", "task_id": task_id}
```

### 3.2 AI Agent Core with LangGraph

```python
# app/agent/graph.py
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import PostgresSaver
from langgraph.prebuilt import ToolExecutor
import json

class AgentState(TypedDict):
    """
    State definition for the Jira AI Agent workflow
    """
    messages: Annotated[list, operator.add]
    user_id: str
    session_id: str
    task_id: str
    jira_context: dict  # Contains ticket info, project context
    llm_response: dict
    next_step: Literal["analyze", "create", "assess", "confirm", "complete"]
    requires_confirmation: bool
    confirmation_data: dict
    current_tool: str
    tool_outputs: list

class JiraAIAgent:
    def __init__(self, config):
        self.config = config
        self.llm = self._initialize_llm()
        self.tools = self._initialize_tools()
        self.graph = self._build_graph()
        
    def _initialize_llm(self):
        """Initialize LLM with dynamic API key management"""
        from app.llm.manager import OpenAIManager
        
        return OpenAIManager(
            model_name="gpt-4.1",
            temperature=0.1,
            max_tokens=2000
        )
    
    def _initialize_tools(self):
        """Initialize MCP tools for Jira operations"""
        from app.tools.jira_tools import (
            create_jira_ticket,
            get_jira_ticket,
            search_jira_tickets,
            update_jira_ticket,
            analyze_requirements
        )
        
        return {
            "create_jira_ticket": create_jira_ticket,
            "get_jira_ticket": get_jira_ticket,
            "search_jira_tickets": search_jira_tickets,
            "update_jira_ticket": update_jira_ticket,
            "analyze_requirements": analyze_requirements
        }
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("plan_actions", self.plan_actions)
        workflow.add_node("execute_tools", self.execute_tools)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("check_confirmation", self.check_confirmation)
        workflow.add_node("handle_confirmation", self.handle_confirmation)
        
        # Define edges
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_by_intent,
            {
                "create_ticket": "plan_actions",
                "assess_ticket": "plan_actions",
                "analyze_requirements": "generate_response",
                "unknown": "generate_response"
            }
        )
        
        workflow.add_edge("plan_actions", "execute_tools")
        
        workflow.add_conditional_edges(
            "execute_tools",
            self.check_tool_output,
            {
                "needs_confirmation": "check_confirmation",
                "complete": "generate_response"
            }
        )
        
        workflow.add_conditional_edges(
            "check_confirmation",
            self.route_confirmation,
            {
                "confirmed": "execute_tools",
                "needs_input": "handle_confirmation",
                "cancelled": "generate_response"
            }
        )
        
        workflow.add_edge("handle_confirmation", "execute_tools")
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add checkpointing
        checkpointer = PostgresSaver.from_conn_string(
            self.config.postgres_uri,
            serde="json"
        )
        
        return workflow.compile(checkpointer=checkpointer)
    
    def classify_intent(self, state: AgentState) -> dict:
        """Classify user intent using LLM"""
        prompt = self.prompt_manager.get_prompt("intent_classification")
        messages = prompt.format_messages(
            user_message=state["messages"][-1].content,
            history=state["messages"][:-1]
        )
        
        response = self.llm.invoke(messages)
        intent = self._parse_intent(response.content)
        
        return {"next_step": intent, "llm_response": response.dict()}
    
    def plan_actions(self, state: AgentState) -> dict:
        """Plan the sequence of actions needed"""
        prompt = self.prompt_manager.get_prompt("action_planning")
        
        if state["next_step"] == "create_ticket":
            messages = prompt.format_messages(
                requirement=state["messages"][-1].content,
                context=state.get("jira_context", {})
            )
        else:  # assess_ticket
            messages = prompt.format_messages(
                ticket_number=state["jira_context"].get("ticket_number"),
                context=state["jira_context"]
            )
        
        response = self.llm.invoke(messages)
        plan = json.loads(response.content)
        
        return {
            "current_tool": plan["first_tool"],
            "tool_sequence": plan["sequence"]
        }
    
    def execute_tools(self, state: AgentState) -> dict:
        """Execute the current tool in the sequence"""
        tool_name = state["current_tool"]
        tool = self.tools[tool_name]
        
        # Prepare tool inputs from state
        tool_inputs = self._prepare_tool_inputs(state, tool_name)
        
        # Execute tool
        result = tool(**tool_inputs)
        
        # Check if confirmation is needed
        requires_confirmation = result.get("requires_confirmation", False)
        
        return {
            "tool_outputs": state["tool_outputs"] + [result],
            "requires_confirmation": requires_confirmation,
            "confirmation_data": result.get("confirmation_data", {}),
            "current_tool": self._get_next_tool(state)
        }
    
    def check_confirmation(self, state: AgentState) -> dict:
        """Check if we need to wait for user confirmation"""
        if state.get("requires_confirmation"):
            # Pause the workflow and wait for user input
            raise InterruptionRequired(
                "waiting_for_confirmation",
                data=state["confirmation_data"]
            )
        return {"next_step": "complete"}
    
    def handle_confirmation(self, state: AgentState) -> dict:
        """Handle user confirmation input"""
        # This node is only reached after resuming from interruption
        user_input = state.get("resume_data", {}).get("user_input")
        
        if user_input.lower() in ["yes", "confirm", "proceed"]:
            return {"next_step": "confirmed", "requires_confirmation": False}
        else:
            return {"next_step": "cancelled", "requires_confirmation": False}
    
    def generate_response(self, state: AgentState) -> dict:
        """Generate final response to user"""
        prompt = self.prompt_manager.get_prompt("response_generation")
        
        messages = prompt.format_messages(
            tool_outputs=state["tool_outputs"],
            original_query=state["messages"][-1].content,
            context=state["jira_context"]
        )
        
        response = self.llm.invoke(messages)
        
        # Format response based on request
        if state.get("format") == "json":
            output = self._format_json_response(response)
        else:
            output = self._format_markdown_response(response)
        
        return {
            "response": output,
            "next_step": "complete"
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
        
    except Exception as e:
        self.on_failure(e, task_id, None, None, None)
        raise

@celery_app.task(base=AgentTask, bind=True, name="resume_agent_workflow")
def resume_agent_workflow(self, task_id, user_input, parameters):
    """Resume a paused workflow"""
    self.initialize_agent()
    
    # Get the paused state
    from app.storage.checkpoint_store import CheckpointStore
    
    checkpoint_store = CheckpointStore()
    state = checkpoint_store.get_state(task_id)
    
    if not state:
        raise ValueError(f"No paused state found for task {task_id}")
    
    # Add user input to state
    state["resume_data"] = {
        "user_input": user_input,
        "parameters": parameters
    }
    
    # Resume execution
    config = {"configurable": {"thread_id": state["session_id"]}}
    
    for event in self.agent.graph.stream(state, config, as_state=True):
        # Continue processing...
        pass
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
        name: intent_classification
        version: 1.0
        variables:
          - user_message
          - history
        ---
        
        # System Prompt
        
        {{ instruction }}
        
        ## Examples
        
        {{ examples }}
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
        if prompt_name not in self.prompt_cache:
            # Try to fetch from config center
            self._fetch_from_config_center(prompt_name)
        
        prompt_data = self.prompt_cache[prompt_name]
        
        if version and prompt_data["metadata"].get("version") != version:
            # Fetch specific version
            self._fetch_from_config_center(prompt_name, version)
            prompt_data = self.prompt_cache[f"{prompt_name}_v{version}"]
        
        return PromptTemplate(
            name=prompt_name,
            template=prompt_data["template"],
            variables=prompt_data["metadata"].get("variables", [])
        )
    
    def _fetch_from_config_center(self, prompt_name: str, version: str = None):
        """Fetch prompt from external config center"""
        from app.config.service import ConfigService
        
        config_service = ConfigService()
        prompt_content = config_service.get_prompt(prompt_name, version)
        
        if prompt_content:
            prompt_data = self._parse_prompt_markdown(prompt_content)
            cache_key = f"{prompt_name}_v{version}" if version else prompt_name
            self.prompt_cache[cache_key] = prompt_data

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
    [*] --> ClassifyIntent
    
    state ClassifyIntent {
        [*] --> ExtractEntities
        ExtractEntities --> DetermineIntent
        DetermineIntent --> [*]
    }
    
    ClassifyIntent --> PlanActions : create_ticket/assess_ticket
    ClassifyIntent --> GenerateResponse : analyze_requirements
    
    state PlanActions {
        [*] --> AnalyzeRequirements
        AnalyzeRequirements --> GenerateToolSequence
        GenerateToolSequence --> [*]
    }
    
    PlanActions --> ExecuteTools
    
    state ExecuteTools {
        [*] --> ValidateInputs
        ValidateInputs --> CallJiraAPI
        CallJiraAPI --> CheckResult
        CheckResult --> [*]
    }
    
    ExecuteTools --> CheckConfirmation : needs_confirmation
    ExecuteTools --> GenerateResponse : complete
    
    state CheckConfirmation {
        [*] --> EvaluateRisk
        EvaluateRisk --> SetInterruption
        SetInterruption --> [*]
    }
    
    CheckConfirmation --> GenerateResponse : cancelled
    CheckConfirmation --> PausedState : needs_input
    
    state PausedState {
        [*] --> WaitForUser
        WaitForUser --> ValidateInput
        ValidateInput --> [*]
    }
    
    PausedState --> HandleConfirmation
    
    HandleConfirmation --> ExecuteTools : confirmed
    
    state GenerateResponse {
        [*] --> FormatOutput
        FormatOutput --> JSON/Markdown
        JSON/Markdown --> [*]
    }
    
    GenerateResponse --> [*]
```

## 5. Data Models

```python
# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TicketPriority(str, Enum):
    LOWEST = "Lowest"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    HIGHEST = "Highest"

class TicketType(str, Enum):
    BUG = "Bug"
    TASK = "Task"
    STORY = "Story"
    EPIC = "Epic"
    IMPROVEMENT = "Improvement"

class JiraTicketCreate(BaseModel):
    """Schema for creating a Jira ticket"""
    project_key: str = Field(..., description="Jira project key")
    summary: str = Field(..., max_length=255)
    description: str
    issue_type: TicketType = TicketType.TASK
    priority: TicketPriority = TicketPriority.MEDIUM
    assignee: Optional[str] = None
    labels: List[str] = []
    custom_fields: Dict[str, Any] = {}

class AgentResponse(BaseModel):
    """Schema for agent response"""
    task_id: str
    session_id: str
    status: str  # processing, completed, paused, failed
    response: Optional[Dict[str, Any]] = None
    formatted_response: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: Optional[datetime] = None

class ConversationState(BaseModel):
    """Schema for conversation state in checkpoints"""
    thread_id: str
    user_id: str
    messages: List[Dict[str, Any]]
    context: Dict[str, Any]
    tool_state: Dict[str, Any]
    next_action: Optional[str] = None
    requires_input: bool = False
    last_updated: datetime

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

### 6.1 Use Case 1: Create Jira Ticket from Natural Language

**User Input:**
```
"I need a new bug ticket for the login page. The submit button disappears on mobile devices. 
Priority is high and assign to John in the WEB project."
```

**Agent Workflow:**
1. **Intent Classification**: Identifies as ticket creation
2. **Entity Extraction**: 
   - Project: WEB
   - Issue Type: Bug
   - Summary: Login page submit button disappears on mobile
   - Priority: High
   - Assignee: John
3. **Action Planning**: Plans to validate project, extract missing fields, create ticket
4. **Tool Execution**: 
   - Calls Jira API to validate project WEB exists
   - Asks for missing description details if needed
   - Creates ticket with structured data
5. **Response Generation**: Returns ticket link and summary

**Output (Markdown):**
```markdown
✅ **Ticket Created Successfully**

**Ticket**: [WEB-1234](https://your-jira.com/browse/WEB-1234)
**Summary**: Login page submit button disappears on mobile devices
**Type**: Bug
**Priority**: High
**Assignee**: John Doe
**Status**: To Do

**Additional Details**:
- Created in project: Web Development (WEB)
- Labels: frontend, mobile, bug
- Estimated fix time: 2-3 hours
```

### 6.2 Use Case 2: Ticket Assessment and Analysis

**User Input:**
```
"Analyze ticket PROJ-567 and suggest next steps. Is it blocked?"
```

**Agent Workflow:**
1. **Intent Classification**: Identifies as ticket assessment
2. **Data Gathering**: 
   - Fetches PROJ-567 from Jira
   - Retrieves related tickets, comments, attachments
   - Gets recent activity
3. **Analysis**:
   - Uses LLM to analyze ticket status, comments, relationships
   - Identifies blockers and dependencies
   - Suggests next actions
4. **Response Generation**: Structured assessment report

**Output (JSON):**
```json
{
  "ticket": "PROJ-567",
  "status": "In Progress",
  "blockers": [
    {
      "type": "dependency",
      "ticket": "PROJ-543",
      "status": "In Review",
      "description": "Waiting for API changes"
    }
  ],
  "analysis": {
    "complexity": "Medium",
    "risk": "Low",
    "estimated_completion": "3 days",
    "suggested_actions": [
      "Follow up with PROJ-543 assignee",
      "Update documentation",
      "Schedule QA testing"
    ]
  },
  "related_tickets": ["PROJ-543", "PROJ-589"],
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### 6.3 Use Case 3: Human-in-the-Loop Confirmation

**Scenario**: Agent needs approval before making significant changes

**Workflow Pause:**
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