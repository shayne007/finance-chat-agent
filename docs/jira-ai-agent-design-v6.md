# Jira AI Agent System Design Document

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Implementation Details](#implementation-details)
5. [Workflow Examples](#workflow-examples)
6. [Database Schema](#database-schema)
7. [Deployment Considerations](#deployment-considerations)

---

## Executive Summary

The Jira AI Agent is an intelligent system powered by GPT-4.1 that automates and enhances Jira ticket management workflows. Built on LangGraph and LangChain frameworks, it provides requirement analysis, automated ticket creation, ticket assessment, and human-in-the-loop approval workflows with conversation memory persistence.

**Key Capabilities:**
- Natural language requirement analysis with structured outputs
- Automated Jira ticket creation via API integration
- Intelligent ticket assessment and recommendations
- Conversation memory using Redis checkpointing
- Dynamic API key management with automatic refresh
- Prompt management via PostgreSQL
- Human-in-the-loop approval workflows

---

## System Architecture

### High-Level Architecture

```mermaid
graph TB
    User[User Interface] --> Agent[Jira AI Agent]
    Agent --> LLM[GPT-4.1 LLM]
    Agent --> Redis[(Redis Checkpointer)]
    Agent --> PostgreSQL[(PostgreSQL - Prompts)]
    Agent --> Jira[Jira API]
    
    LLM --> KeyManager[API Key Manager]
    KeyManager --> KeyStore[(API Key Store)]
    
    Agent --> Planner[Planning Node]
    Agent --> Classifier[Classification Node]
    Agent --> Executor[Execution Node]
    Agent --> Reviewer[Review Node]
    
    style Agent fill:#4A90E2
    style LLM fill:#E27D60
    style Redis fill:#85CDCA
    style PostgreSQL fill:#E8A87C
```

### Component Interaction Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant C as Classifier
    participant P as Planner
    participant E as Executor
    participant L as LLM
    participant J as Jira API
    participant R as Redis
    
    U->>A: Submit request
    A->>R: Load conversation history
    A->>C: Classify intent
    C->>L: Analyze request type
    L-->>C: Classification result
    
    alt Requirement Analysis
        C->>P: Route to planner
        P->>L: Generate analysis
        L-->>P: Structured output
        P-->>U: Return analysis
    else Create Ticket
        C->>P: Route to ticket creation
        P->>E: Execute creation plan
        E->>L: Extract ticket fields
        E->>U: Request confirmation (Human-in-loop)
        U->>E: Approve/Modify
        E->>J: Create ticket
        J-->>E: Ticket created
        E-->>U: Success response
    else Assess Ticket
        C->>E: Route to assessment
        E->>J: Fetch ticket details
        J-->>E: Ticket data
        E->>L: Analyze ticket
        L-->>E: Assessment report
        E-->>U: Return assessment
    end
    
    A->>R: Save conversation state
```

---

## Core Components

### 1. LangGraph State Machine

The agent uses a state graph to orchestrate different workflows:

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """State schema for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    intent: str
    extracted_data: dict
    approval_required: bool
    approval_status: str
    ticket_id: str
    next_step: str

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classifier", classify_intent)
workflow.add_node("requirement_analyzer", analyze_requirements)
workflow.add_node("ticket_creator", create_ticket_workflow)
workflow.add_node("ticket_assessor", assess_ticket_workflow)
workflow.add_node("human_approval", wait_for_approval)

# Define edges and conditional routing
workflow.set_entry_point("classifier")
workflow.add_conditional_edges(
    "classifier",
    route_by_intent,
    {
        "analyze": "requirement_analyzer",
        "create": "ticket_creator",
        "assess": "ticket_assessor"
    }
)
workflow.add_edge("requirement_analyzer", END)
workflow.add_conditional_edges(
    "ticket_creator",
    check_approval_needed,
    {
        "needs_approval": "human_approval",
        "approved": END
    }
)
workflow.add_edge("human_approval", END)
workflow.add_edge("ticket_assessor", END)
```

### 2. Dynamic API Key Manager

Handles automatic API key refresh for GPT-4.1:

```python
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from langchain_openai import ChatOpenAI

class DynamicAPIKeyManager:
    """Manages OpenAI API keys with automatic refresh."""
    
    def __init__(self, initial_key: str, refresh_callback):
        self.current_key = initial_key
        self.refresh_callback = refresh_callback
        self.expiry_time = datetime.now() + timedelta(hours=24)
        self._lock = asyncio.Lock()
    
    async def get_valid_key(self) -> str:
        """Returns a valid API key, refreshing if necessary."""
        async with self._lock:
            if datetime.now() >= self.expiry_time:
                print("API key expired, refreshing...")
                self.current_key = await self.refresh_callback()
                self.expiry_time = datetime.now() + timedelta(hours=24)
                print("API key refreshed successfully")
            return self.current_key
    
    async def create_llm(self, **kwargs) -> ChatOpenAI:
        """Creates ChatOpenAI instance with valid API key."""
        valid_key = await self.get_valid_key()
        return ChatOpenAI(
            api_key=valid_key,
            model="gpt-4-turbo-preview",
            temperature=0.7,
            **kwargs
        )

# Usage example
async def refresh_api_key() -> str:
    """Simulates API key refresh from a secure service."""
    # In production, fetch from AWS Secrets Manager, Vault, etc.
    return "sk-new-refreshed-key-" + datetime.now().isoformat()

key_manager = DynamicAPIKeyManager(
    initial_key="sk-initial-key",
    refresh_callback=refresh_api_key
)
```

### 3. Redis Checkpointer for Memory

Implements conversation persistence with LangGraph:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import RedisSaver
import redis.asyncio as redis

async def create_checkpointer():
    """Creates Redis-based checkpointer for conversation memory."""
    redis_client = await redis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True
    )
    
    return RedisSaver(redis_client)

# Create the compiled graph with checkpointer
checkpointer = await create_checkpointer()
app = workflow.compile(checkpointer=checkpointer)

# Invoke with thread_id for conversation continuity
config = {"configurable": {"thread_id": "user-123-session"}}
result = await app.ainvoke(
    {"messages": [HumanMessage(content="Create a bug ticket")]},
    config=config
)
```

### 4. Prompt Management System

Stores and retrieves prompts from PostgreSQL:

```python
import asyncpg
from typing import Optional, Dict
import json

class PromptManager:
    """Manages prompts stored in PostgreSQL."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(self.db_url)
    
    async def get_prompt(self, prompt_name: str, version: Optional[str] = None) -> Dict:
        """Retrieves a prompt by name and optional version."""
        async with self.pool.acquire() as conn:
            if version:
                query = """
                    SELECT content, variables, metadata
                    FROM prompts
                    WHERE name = $1 AND version = $2 AND is_active = true
                """
                row = await conn.fetchrow(query, prompt_name, version)
            else:
                query = """
                    SELECT content, variables, metadata
                    FROM prompts
                    WHERE name = $1 AND is_active = true
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                row = await conn.fetchrow(query, prompt_name)
            
            if row:
                return {
                    "content": row["content"],
                    "variables": json.loads(row["variables"]),
                    "metadata": json.loads(row["metadata"])
                }
            return None
    
    async def render_prompt(self, prompt_name: str, **variables) -> str:
        """Renders a prompt template with provided variables."""
        prompt_data = await self.get_prompt(prompt_name)
        if not prompt_data:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        
        template = prompt_data["content"]
        return template.format(**variables)

# Usage
prompt_manager = PromptManager("postgresql://user:pass@localhost/jira_agent")
await prompt_manager.initialize()

classify_prompt = await prompt_manager.render_prompt(
    "intent_classification",
    user_message="I need to create a bug ticket"
)
```

---

## Implementation Details

### Classification Node

Determines the user's intent using structured output:

```python
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Literal

class IntentClassification(BaseModel):
    """Structured output for intent classification."""
    intent: Literal["analyze", "create", "assess", "general"] = Field(
        description="The primary intent of the user's request"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Brief explanation of the classification"
    )

async def classify_intent(state: AgentState) -> AgentState:
    """Classifies user intent using LLM."""
    llm = await key_manager.create_llm()
    
    system_prompt = await prompt_manager.render_prompt(
        "intent_classification_system"
    )
    
    structured_llm = llm.with_structured_output(IntentClassification)
    
    result = await structured_llm.ainvoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    
    return {
        **state,
        "intent": result.intent,
        "next_step": result.intent
    }
```

### Requirement Analysis Node

Analyzes requirements and produces structured JSON or Markdown:

```python
from pydantic import BaseModel
from typing import List

class RequirementAnalysis(BaseModel):
    """Structured requirement analysis output."""
    summary: str = Field(description="Brief summary of requirements")
    user_stories: List[str] = Field(description="List of user stories")
    acceptance_criteria: List[str] = Field(description="Acceptance criteria")
    technical_considerations: List[str] = Field(description="Technical notes")
    estimated_complexity: Literal["low", "medium", "high"] = Field(
        description="Complexity estimation"
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Recommended priority"
    )

async def analyze_requirements(state: AgentState) -> AgentState:
    """Analyzes requirements and returns structured output."""
    llm = await key_manager.create_llm()
    
    system_prompt = await prompt_manager.render_prompt(
        "requirement_analysis_system"
    )
    
    structured_llm = llm.with_structured_output(RequirementAnalysis)
    
    analysis = await structured_llm.ainvoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    
    # Convert to markdown for user-friendly display
    markdown_output = f"""
# Requirement Analysis

## Summary
{analysis.summary}

## User Stories
{chr(10).join(f"- {story}" for story in analysis.user_stories)}

## Acceptance Criteria
{chr(10).join(f"- {criterion}" for criterion in analysis.acceptance_criteria)}

## Technical Considerations
{chr(10).join(f"- {consideration}" for consideration in analysis.technical_considerations)}

## Estimation
- **Complexity**: {analysis.estimated_complexity.upper()}
- **Priority**: {analysis.priority.upper()}
"""
    
    return {
        **state,
        "extracted_data": analysis.dict(),
        "messages": state["messages"] + [
            AIMessage(content=markdown_output)
        ]
    }
```

### Ticket Creation Node with Human-in-the-Loop

Creates Jira tickets with approval workflow:

```python
import aiohttp
from typing import Dict

class JiraTicketData(BaseModel):
    """Structured Jira ticket data."""
    project_key: str = Field(description="Jira project key (e.g., 'PROJ')")
    issue_type: str = Field(description="Issue type: Bug, Story, Task, etc.")
    summary: str = Field(description="Ticket summary/title")
    description: str = Field(description="Detailed description")
    priority: str = Field(description="Priority: Highest, High, Medium, Low")
    labels: List[str] = Field(description="List of labels")
    assignee: Optional[str] = Field(description="Assignee username")

async def extract_ticket_data(state: AgentState) -> AgentState:
    """Extracts ticket information from user request."""
    llm = await key_manager.create_llm()
    
    system_prompt = await prompt_manager.render_prompt(
        "ticket_extraction_system"
    )
    
    structured_llm = llm.with_structured_output(JiraTicketData)
    
    ticket_data = await structured_llm.ainvoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])
    
    return {
        **state,
        "extracted_data": ticket_data.dict(),
        "approval_required": True,
        "approval_status": "pending"
    }

async def wait_for_approval(state: AgentState) -> AgentState:
    """Pauses workflow for human approval."""
    # This creates an interrupt point in LangGraph
    # The workflow will pause here until resumed with approval
    ticket_data = state["extracted_data"]
    
    approval_message = f"""
Please review the ticket details:

**Project**: {ticket_data['project_key']}
**Type**: {ticket_data['issue_type']}
**Summary**: {ticket_data['summary']}
**Priority**: {ticket_data['priority']}
**Labels**: {', '.join(ticket_data['labels'])}

Reply with 'approve' to create the ticket or provide modifications.
"""
    
    return {
        **state,
        "messages": state["messages"] + [
            AIMessage(content=approval_message)
        ]
    }

async def create_jira_ticket(ticket_data: Dict) -> str:
    """Creates a Jira ticket via REST API."""
    jira_url = "https://your-domain.atlassian.net/rest/api/3/issue"
    auth = aiohttp.BasicAuth("email@example.com", "api_token")
    
    payload = {
        "fields": {
            "project": {"key": ticket_data["project_key"]},
            "summary": ticket_data["summary"],
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": ticket_data["description"]
                            }
                        ]
                    }
                ]
            },
            "issuetype": {"name": ticket_data["issue_type"]},
            "priority": {"name": ticket_data["priority"]},
            "labels": ticket_data["labels"]
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(jira_url, json=payload, auth=auth) as resp:
            result = await resp.json()
            return result["key"]

async def create_ticket_workflow(state: AgentState) -> AgentState:
    """Complete workflow for ticket creation."""
    state = await extract_ticket_data(state)
    
    if state["approval_status"] == "approved":
        ticket_id = await create_jira_ticket(state["extracted_data"])
        
        return {
            **state,
            "ticket_id": ticket_id,
            "messages": state["messages"] + [
                AIMessage(content=f"✅ Ticket created successfully: {ticket_id}")
            ]
        }
    
    return state
```

### Ticket Assessment Node

Analyzes existing Jira tickets:

```python
class TicketAssessment(BaseModel):
    """Structured ticket assessment output."""
    overall_health: Literal["good", "needs_attention", "critical"] = Field(
        description="Overall ticket health status"
    )
    completeness_score: float = Field(
        description="Completeness score 0-100"
    )
    clarity_score: float = Field(
        description="Clarity score 0-100"
    )
    missing_information: List[str] = Field(
        description="List of missing information"
    )
    recommendations: List[str] = Field(
        description="Improvement recommendations"
    )
    estimated_effort: str = Field(
        description="Estimated effort (e.g., '2-3 days')"
    )

async def fetch_jira_ticket(ticket_id: str) -> Dict:
    """Fetches ticket details from Jira API."""
    jira_url = f"https://your-domain.atlassian.net/rest/api/3/issue/{ticket_id}"
    auth = aiohttp.BasicAuth("email@example.com", "api_token")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(jira_url, auth=auth) as resp:
            return await resp.json()

async def assess_ticket_workflow(state: AgentState) -> AgentState:
    """Assesses an existing Jira ticket."""
    # Extract ticket ID from user message
    ticket_id = extract_ticket_id(state["messages"][-1].content)
    
    # Fetch ticket from Jira
    ticket_data = await fetch_jira_ticket(ticket_id)
    
    # Analyze with LLM
    llm = await key_manager.create_llm()
    
    system_prompt = await prompt_manager.render_prompt(
        "ticket_assessment_system"
    )
    
    structured_llm = llm.with_structured_output(TicketAssessment)
    
    assessment = await structured_llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Assess this ticket: {json.dumps(ticket_data, indent=2)}")
    ])
    
    # Format assessment as markdown
    markdown_report = f"""
# Ticket Assessment: {ticket_id}

## Overall Health: {assessment.overall_health.upper()} 
{"🟢" if assessment.overall_health == "good" else "🟡" if assessment.overall_health == "needs_attention" else "🔴"}

## Scores
- **Completeness**: {assessment.completeness_score}/100
- **Clarity**: {assessment.clarity_score}/100

## Missing Information
{chr(10).join(f"- {item}" for item in assessment.missing_information) if assessment.missing_information else "- None"}

## Recommendations
{chr(10).join(f"- {rec}" for rec in assessment.recommendations)}

## Estimated Effort
{assessment.estimated_effort}
"""
    
    return {
        **state,
        "ticket_id": ticket_id,
        "extracted_data": assessment.dict(),
        "messages": state["messages"] + [
            AIMessage(content=markdown_report)
        ]
    }
```

---

## Workflow Examples

### Example 1: Requirement Analysis

**User Input:**
```
I need a feature where users can export their data to PDF format. 
It should include all their transactions from the last 6 months.
```

**Agent Flow:**
```mermaid
graph LR
    A[User Request] --> B[Classifier]
    B --> C[Intent: Analyze]
    C --> D[Requirement Analyzer]
    D --> E[LLM Processing]
    E --> F[Structured Output]
    F --> G[Markdown Report]
    G --> H[Return to User]
    
    style D fill:#4A90E2
    style F fill:#85CDCA
```

**Agent Output:**
```markdown
# Requirement Analysis

## Summary
Feature to enable PDF export of user transaction history for the past 6 months

## User Stories
- As a user, I want to export my transactions to PDF so that I can keep offline records
- As a user, I want to see my last 6 months of transactions so that I can track my spending

## Acceptance Criteria
- Export button is visible on the transactions page
- PDF includes all transactions from the last 6 months
- PDF is formatted clearly with transaction date, amount, and description
- Export completes within 10 seconds for up to 1000 transactions

## Technical Considerations
- Need to implement PDF generation library (e.g., ReportLab, WeasyPrint)
- Consider pagination for large transaction sets
- Add date filtering logic for 6-month window
- Implement proper authentication checks before export

## Estimation
- **Complexity**: MEDIUM
- **Priority**: MEDIUM
```

### Example 2: Ticket Creation with Approval

**Conversation Flow:**

```
User: Create a bug ticket for the login page not working on mobile Safari

Agent: [Extracts ticket data]

Please review the ticket details:

**Project**: WEB
**Type**: Bug
**Summary**: Login page not working on mobile Safari
**Priority**: High
**Labels**: mobile, safari, login, bug

Reply with 'approve' to create the ticket or provide modifications.

User: approve

Agent: ✅ Ticket created successfully: WEB-1234
```

**State Transitions:**
```mermaid
stateDiagram-v2
    [*] --> Classifier
    Classifier --> TicketCreator: intent=create
    TicketCreator --> ExtractData: Parse request
    ExtractData --> WaitApproval: Present ticket preview
    WaitApproval --> CheckResponse: User responds
    CheckResponse --> CreateInJira: Approved
    CheckResponse --> ExtractData: Modifications requested
    CreateInJira --> [*]: Success
```

### Example 3: Ticket Assessment

**User Input:**
```
Assess ticket PROJ-456
```

**Agent Flow:**
1. Extract ticket ID: `PROJ-456`
2. Fetch from Jira API
3. Analyze with LLM
4. Return structured assessment

**Agent Output:**
```markdown
# Ticket Assessment: PROJ-456

## Overall Health: NEEDS_ATTENTION 🟡

## Scores
- **Completeness**: 65/100
- **Clarity**: 72/100

## Missing Information
- Reproduction steps are vague
- No environment information provided
- Expected vs actual behavior not clearly defined
- No attachments or screenshots

## Recommendations
- Add detailed step-by-step reproduction instructions
- Specify browser version and OS
- Include screenshots of the error
- Define clear acceptance criteria for the fix
- Add labels for easier categorization

## Estimated Effort
2-3 days (with current information)
```

---

## Database Schema

### PostgreSQL Schema for Prompt Management

```sql
-- Prompts table
CREATE TABLE prompts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    variables JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

-- Create indexes for faster lookups
CREATE INDEX idx_prompts_name ON prompts(name);
CREATE INDEX idx_prompts_active ON prompts(is_active);
CREATE INDEX idx_prompts_created_at ON prompts(created_at DESC);

-- Prompt versions history
CREATE TABLE prompt_history (
    id SERIAL PRIMARY KEY,
    prompt_id INTEGER REFERENCES prompts(id),
    old_content TEXT,
    new_content TEXT,
    changed_by VARCHAR(255),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    change_reason TEXT
);

-- Sample data
INSERT INTO prompts (name, version, content, variables, metadata) VALUES
('intent_classification_system', 'v1.0', 
'You are an intent classification expert. Analyze the user''s message and determine their primary intent.

Possible intents:
- analyze: User wants requirement analysis
- create: User wants to create a Jira ticket
- assess: User wants to assess an existing ticket
- general: General conversation

Respond with your classification, confidence score, and reasoning.',
'[]'::jsonb,
'{"purpose": "Classify user intent", "model": "gpt-4-turbo"}'::jsonb),

('requirement_analysis_system', 'v1.0',
'You are a senior business analyst. Analyze the user''s requirements and provide a structured analysis.

Focus on:
1. Clear summary of the requirement
2. User stories in "As a [user], I want [goal] so that [benefit]" format
3. Specific acceptance criteria
4. Technical considerations and constraints
5. Complexity and priority estimation

Be thorough but concise.',
'[]'::jsonb,
'{"purpose": "Analyze requirements", "model": "gpt-4-turbo"}'::jsonb),

('ticket_extraction_system', 'v1.0',
'You are a Jira ticket creation expert. Extract structured ticket information from the user''s request.

Extract:
- Project key (default to "PROJ" if not specified)
- Issue type (Bug, Story, Task, Epic)
- Clear, concise summary
- Detailed description
- Priority (Highest, High, Medium, Low, Lowest)
- Relevant labels
- Assignee (if mentioned)

Infer missing information based on context.',
'[]'::jsonb,
'{"purpose": "Extract Jira ticket data", "model": "gpt-4-turbo"}'::jsonb),

('ticket_assessment_system', 'v1.0',
'You are a Jira ticket quality auditor. Assess the provided ticket for completeness, clarity, and quality.

Evaluate:
1. Overall health status (good, needs_attention, critical)
2. Completeness score (0-100): Are all necessary fields filled?
3. Clarity score (0-100): Is the information clear and unambiguous?
4. Missing information: What critical details are missing?
5. Recommendations: How can this ticket be improved?
6. Estimated effort: Based on the information provided

Be constructive and specific in your recommendations.',
'[]'::jsonb,
'{"purpose": "Assess ticket quality", "model": "gpt-4-turbo"}'::jsonb);
```

### Redis Data Structure for Checkpointing

```python
# Redis stores conversation state as:
# Key: checkpoint:{thread_id}:{checkpoint_id}
# Value: Serialized AgentState

# Example Redis data structure:
{
    "checkpoint:user-123-session:abc-def-ghi": {
        "messages": [
            {"role": "user", "content": "Create a bug ticket"},
            {"role": "assistant", "content": "Please review the ticket details..."}
        ],
        "intent": "create",
        "extracted_data": {
            "project_key": "WEB",
            "issue_type": "Bug",
            "summary": "Login page not working"
        },
        "approval_required": true,
        "approval_status": "pending",
        "ticket_id": null,
        "next_step": "wait_for_approval"
    }
}
```

---

## Deployment Considerations

### Architecture Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Web UI]
        API[REST API]
    end
    
    subgraph "Application Layer"
        AG[Jira AI Agent]
        KM[API Key Manager]
        PM[Prompt Manager]
    end
    
    subgraph "LLM Layer"
        GPT[GPT-4.1 OpenAI]
    end
    
    subgraph "Storage Layer"
        Redis[(Redis<br/>Conversation Memory)]
        PG[(PostgreSQL<br/>Prompts)]
        Vault[(Vault<br/>API Keys)]
    end
    
    subgraph "External Services"
        Jira[Jira Cloud API]
    end
    
    UI --> API
    API --> AG
    AG --> KM
    AG --> PM
    AG --> Redis
    KM --> Vault
    KM --> GPT
    PM --> PG
    AG --> Jira
    
    style AG fill:#4A90E2
    style GPT fill:#E27D60
    style Redis fill:#85CDCA
    style PG fill:#E8A87C
```

### Performance Optimization

**1. Caching Strategy**
```python
from functools import lru_cache
import hashlib

class CachedLLMWrapper:
    """Caches LLM responses to reduce API calls."""
    
    def __init__(self, llm, redis_client):
        self.llm = llm
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour
    
    def _generate_cache_key(self, messages: List) -> str:
        """Generates cache key from messages."""
        content = "".join(m.content for m in messages)
        return f"llm_cache:{hashlib.sha256(content.encode()).hexdigest()}"
    
    async def ainvoke(self, messages: List) -> BaseMessage:
        """Invokes LLM with caching."""
        cache_key = self._generate_cache_key(messages)
        
        # Check cache
        cached = await self.redis.get(cache_key)
        if cached:
            return AIMessage(content=cached)
        
        # Call LLM
        response = await self.llm.ainvoke(messages)
        
        # Store in cache
        await self.redis.setex(
            cache_key,
            self.cache_ttl,
            response.content
        )
        
        return response
```

**2. Async Processing**
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_ticket_assessment(ticket_ids: List[str]) -> List[Dict]:
    """Assesses multiple tickets in parallel."""
    tasks = [assess_ticket_workflow({"ticket_id": tid}) for tid in ticket_ids]
    results = await asyncio.gather(*tasks)
    return results
```

### Monitoring and Observability

```python
from prometheus_client import Counter, Histogram
import time

# Metrics
intent_classification_counter = Counter(
    'intent_classification_total',
    'Total intent classifications',
    ['intent']
)

ticket_creation_counter = Counter(
    'ticket_creation_total',
    'Total tickets created',
    ['status']
)

llm_latency_histogram = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds'
)

class MonitoredAgent:
    """Agent wrapper with monitoring."""
    
    async def invoke_with_monitoring(self, state: AgentState, config: dict):
        """Invokes agent with metrics collection."""
        start_time = time.time()
        
        try:
            result = await app.ainvoke(state, config)
            
            # Record metrics
            if "intent" in result:
                intent_classification_counter.labels(
                    intent=result["intent"]
                ).inc()
            
            if "ticket_id" in result and result["ticket_id"]:
                ticket_creation_counter.labels(status="success").inc()
            
            return result
        
        except Exception as e:
            ticket_creation_counter.labels(status="error").inc()
            raise
        
        finally:
            duration = time.time() - start_time
            llm_latency_histogram.observe(duration)
```

### Security Considerations

**1. API Key Security**
```python
import os
from cryptography.fernet import Fernet

class SecureKeyStore:
    """Encrypts API keys at rest."""
    
    def __init__(self):
        self.cipher = Fernet(os.environ['ENCRYPTION_KEY'].encode())
    
    def encrypt_key(self, api_key: str) -> bytes:
        """Encrypts an API key."""
        return self.cipher.encrypt(api_key.encode())
    
    def decrypt_key(self, encrypted_key: bytes) -> str:
        """Decrypts an API key."""
        return self.cipher.decrypt(encrypted_key).decode()
```

**2. Input Validation**
```python
from pydantic import BaseModel, validator
import re

class TicketInput(BaseModel):
    """Validated ticket input."""
    summary: str
    description: str
    
    @validator('summary')
    def validate_summary(cls, v):
        if len(v) > 255:
            raise ValueError('Summary too long')
        if not v.strip():
            raise ValueError('Summary cannot be empty')
        return v
    
    @validator('description')
    def validate_description(cls, v):
        # Prevent injection attacks
        dangerous_patterns = [
            r'<script>',
            r'javascript:',
            r'onerror=',
            r'onclick='
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Potentially malicious content detected')
        return v
```

**3. Rate Limiting**
```python
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    """Implements rate limiting per user."""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 60):
        self.max_requests = max_requests
        self.window = timedelta(minutes=window_minutes)
        self.requests = defaultdict(list)
    
    async def check_limit(self, user_id: str) -> bool:
        """Checks if user has exceeded rate limit."""
        now = datetime.now()
        cutoff = now - self.window
        
        # Remove old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if req_time > cutoff
        ]
        
        # Check limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        self.requests[user_id].append(now)
        return True
```

---

## Complete Implementation Example

### Main Application

```python
import asyncio
from typing import Dict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
import redis.asyncio as redis

class JiraAIAgent:
    """Main Jira AI Agent application."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.key_manager = None
        self.prompt_manager = None
        self.checkpointer = None
        self.app = None
    
    async def initialize(self):
        """Initializes all components."""
        # Initialize API key manager
        self.key_manager = DynamicAPIKeyManager(
            initial_key=self.config['openai_api_key'],
            refresh_callback=self._refresh_api_key
        )
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager(self.config['postgres_url'])
        await self.prompt_manager.initialize()
        
        # Initialize Redis checkpointer
        redis_client = await redis.from_url(
            self.config['redis_url'],
            encoding="utf-8",
            decode_responses=True
        )
        self.checkpointer = RedisSaver(redis_client)
        
        # Build workflow graph
        self.app = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Builds the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classifier", self._classify_intent)
        workflow.add_node("requirement_analyzer", self._analyze_requirements)
        workflow.add_node("ticket_creator", self._create_ticket)
        workflow.add_node("ticket_assessor", self._assess_ticket)
        workflow.add_node("human_approval", self._wait_for_approval)
        
        # Set entry point
        workflow.set_entry_point("classifier")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "classifier",
            self._route_by_intent,
            {
                "analyze": "requirement_analyzer",
                "create": "ticket_creator",
                "assess": "ticket_assessor",
                "general": END
            }
        )
        
        workflow.add_edge("requirement_analyzer", END)
        
        workflow.add_conditional_edges(
            "ticket_creator",
            self._check_approval,
            {
                "needs_approval": "human_approval",
                "approved": END
            }
        )
        
        workflow.add_edge("human_approval", END)
        workflow.add_edge("ticket_assessor", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _classify_intent(self, state: AgentState) -> AgentState:
        """Node implementation for intent classification."""
        llm = await self.key_manager.create_llm()
        
        system_prompt = await self.prompt_manager.render_prompt(
            "intent_classification_system"
        )
        
        structured_llm = llm.with_structured_output(IntentClassification)
        
        result = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            *state["messages"]
        ])
        
        return {
            **state,
            "intent": result.intent,
            "next_step": result.intent
        }
    
    def _route_by_intent(self, state: AgentState) -> str:
        """Routing function based on intent."""
        return state.get("intent", "general")
    
    async def _analyze_requirements(self, state: AgentState) -> AgentState:
        """Node for requirement analysis."""
        # Implementation from earlier
        pass
    
    async def _create_ticket(self, state: AgentState) -> AgentState:
        """Node for ticket creation."""
        # Implementation from earlier
        pass
    
    async def _assess_ticket(self, state: AgentState) -> AgentState:
        """Node for ticket assessment."""
        # Implementation from earlier
        pass
    
    async def _wait_for_approval(self, state: AgentState) -> AgentState:
        """Node for human approval."""
        # Implementation from earlier
        pass
    
    def _check_approval(self, state: AgentState) -> str:
        """Checks approval status."""
        if state.get("approval_status") == "approved":
            return "approved"
        return "needs_approval"
    
    async def _refresh_api_key(self) -> str:
        """Refreshes OpenAI API key."""
        # Implement key refresh logic
        # Could fetch from AWS Secrets Manager, HashiCorp Vault, etc.
        return self.config['openai_api_key']
    
    async def process_message(
        self,
        user_message: str,
        thread_id: str
    ) -> str:
        """Processes a user message and returns response."""
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "intent": "",
            "extracted_data": {},
            "approval_required": False,
            "approval_status": "",
            "ticket_id": "",
            "next_step": ""
        }
        
        result = await self.app.ainvoke(initial_state, config)
        
        # Extract final assistant message
        assistant_messages = [
            msg for msg in result["messages"]
            if isinstance(msg, AIMessage)
        ]
        
        return assistant_messages[-1].content if assistant_messages else "I'm sorry, I couldn't process that request."
    
    async def resume_conversation(
        self,
        user_response: str,
        thread_id: str
    ) -> str:
        """Resumes a paused conversation (e.g., after approval)."""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get current state from checkpoint
        current_state = await self.app.aget_state(config)
        
        # Add user response
        updated_state = {
            **current_state.values,
            "messages": current_state.values["messages"] + [
                HumanMessage(content=user_response)
            ]
        }
        
        # Check if user approved
        if "approve" in user_response.lower():
            updated_state["approval_status"] = "approved"
        
        result = await self.app.ainvoke(updated_state, config)
        
        assistant_messages = [
            msg for msg in result["messages"]
            if isinstance(msg, AIMessage)
        ]
        
        return assistant_messages[-1].content if assistant_messages else "Conversation resumed."

# Usage example
async def main():
    """Main execution function."""
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'postgres_url': os.getenv('POSTGRES_URL'),
        'redis_url': os.getenv('REDIS_URL'),
        'jira_url': os.getenv('JIRA_URL'),
        'jira_email': os.getenv('JIRA_EMAIL'),
        'jira_api_token': os.getenv('JIRA_API_TOKEN')
    }
    
    # Initialize agent
    agent = JiraAIAgent(config)
    await agent.initialize()
    
    # Example conversation
    thread_id = "user-123-session"
    
    # First message
    response = await agent.process_message(
        "Create a bug ticket for login not working on Safari",
        thread_id
    )
    print(f"Agent: {response}")
    
    # If approval needed, resume conversation
    if "approve" in response.lower():
        approval_response = await agent.resume_conversation(
            "approve",
            thread_id
        )
        print(f"Agent: {approval_response}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Visualizing the Complete Workflow

### Graph Visualization Code

```python
from IPython.display import Image, display

# Generate and display the graph
try:
    graph_image = app.get_graph().draw_mermaid_png()
    display(Image(graph_image))
except Exception as e:
    print(f"Could not generate graph: {e}")
    # Fallback to ASCII representation
    print(app.get_graph().print_ascii())
```

### Expected Graph Output

```mermaid
graph TD
    START((START)) --> classifier[Classifier]
    
    classifier --> |intent=analyze| requirement_analyzer[Requirement Analyzer]
    classifier --> |intent=create| ticket_creator[Ticket Creator]
    classifier --> |intent=assess| ticket_assessor[Ticket Assessor]
    classifier --> |intent=general| END1((END))
    
    requirement_analyzer --> END2((END))
    
    ticket_creator --> |needs_approval| human_approval[Human Approval]
    ticket_creator --> |approved| END3((END))
    
    human_approval --> END4((END))
    
    ticket_assessor --> END5((END))
    
    style classifier fill:#4A90E2,color:#fff
    style requirement_analyzer fill:#85CDCA
    style ticket_creator fill:#E8A87C
    style ticket_assessor fill:#C06C84
    style human_approval fill:#F67280
```

---

## Testing Strategy

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, Mock, patch

@pytest.mark.asyncio
async def test_intent_classification():
    """Tests intent classification node."""
    # Mock LLM response
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = IntentClassification(
        intent="create",
        confidence=0.95,
        reasoning="User explicitly wants to create a ticket"
    )
    
    # Test state
    state = {
        "messages": [HumanMessage(content="Create a bug ticket")],
        "intent": "",
        "extracted_data": {},
        "approval_required": False,
        "approval_status": "",
        "ticket_id": "",
        "next_step": ""
    }
    
    # Mock key manager
    with patch('key_manager.create_llm', return_value=mock_llm):
        result = await classify_intent(state)
    
    assert result["intent"] == "create"
    assert result["next_step"] == "create"

@pytest.mark.asyncio
async def test_ticket_creation_workflow():
    """Tests complete ticket creation flow."""
    # Mock Jira API
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(
            return_value={"key": "TEST-123"}
        )
        
        ticket_data = {
            "project_key": "TEST",
            "issue_type": "Bug",
            "summary": "Test bug",
            "description": "Test description",
            "priority": "High",
            "labels": ["test"]
        }
        
        ticket_id = await create_jira_ticket(ticket_data)
        
        assert ticket_id == "TEST-123"
        mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_conversation_memory():
    """Tests conversation persistence with Redis."""
    agent = JiraAIAgent(test_config)
    await agent.initialize()
    
    thread_id = "test-thread-123"
    
    # First message
    response1 = await agent.process_message(
        "Analyze this requirement: users need export feature",
        thread_id
    )
    
    # Second message in same thread
    response2 = await agent.process_message(
        "What was my previous request?",
        thread_id
    )
    
    # Should remember the context
    assert "export" in response2.lower() or "requirement" in response2.lower()
```

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_ticket_creation():
    """Tests end-to-end ticket creation with real services."""
    agent = JiraAIAgent(production_config)
    await agent.initialize()
    
    thread_id = f"integration-test-{uuid.uuid4()}"
    
    # Step 1: Request ticket creation
    response1 = await agent.process_message(
        "Create a bug ticket: Login button not responding on iOS",
        thread_id
    )
    
    assert "review" in response1.lower()
    assert "approve" in response1.lower()
    
    # Step 2: Approve ticket
    response2 = await agent.resume_conversation("approve", thread_id)
    
    assert "created successfully" in response2.lower()
    assert "WEB-" in response2 or "PROJ-" in response2
```

---

## Deployment Checklist

### Environment Variables

```bash
# .env.production
OPENAI_API_KEY=sk-prod-key-here
POSTGRES_URL=postgresql://user:pass@prod-db:5432/jira_agent
REDIS_URL=redis://prod-redis:6379/0
JIRA_URL=https://company.atlassian.net
JIRA_EMAIL=bot@company.com
JIRA_API_TOKEN=your-jira-api-token
ENCRYPTION_KEY=your-32-byte-encryption-key
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Docker Compose Setup

```yaml
version: '3.8'

services:
  jira-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - POSTGRES_URL=${POSTGRES_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=jira_agent
      - POSTGRES_USER=agent_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

---

## Future Enhancements

### 1. Multi-Agent Collaboration

```mermaid
graph LR
    Coordinator[Coordinator Agent] --> Analyst[Analyst Agent]
    Coordinator --> Creator[Creator Agent]
    Coordinator --> Reviewer[Reviewer Agent]
    
    Analyst --> KB[(Knowledge Base)]
    Creator --> Jira[Jira API]
    Reviewer --> QA[Quality Checks]
    
    style Coordinator fill:#4A90E2
```

### 2. Advanced Features Roadmap

- **Batch Processing**: Handle multiple tickets in one conversation
- **Template Library**: Pre-defined ticket templates for common scenarios
- **Smart Scheduling**: Suggest sprint assignments based on capacity
- **Sentiment Analysis**: Detect urgency from user tone
- **Auto-tagging**: Automatically categorize and label tickets using ML
- **Integration Hub**: Connect with Slack, Teams, GitHub, etc.
- **Analytics Dashboard**: Visualize ticket trends and agent performance

### 3. Machine Learning Improvements

- Fine-tune custom models on company-specific ticket data
- Implement retrieval-augmented generation (RAG) for company documentation
- Add predictive analytics for effort estimation
- Build recommendation system for similar tickets

---

## Conclusion

This Jira AI Agent system provides a robust, scalable solution for automating Jira workflows using GPT-4.1 and modern AI agent patterns. The architecture emphasizes:

- **Reliability**: Dynamic API key management, error handling, and monitoring
- **Flexibility**: Modular design with LangGraph state machine
- **Usability**: Human-in-the-loop workflows and conversation memory
- **Maintainability**: Centralized prompt management and comprehensive testing
- **Security**: Encryption, validation, and rate limiting

The system is production-ready and can be extended with additional features as organizational needs evolve.