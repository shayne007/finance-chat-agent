# AI Agent Chat System - System Design Document

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Models](#data-models)
5. [API Design](#api-design)
6. [LLM Integration with LangChain](#llm-integration-with-langchain)
7. [Caching Strategy](#caching-strategy)
8. [Use Cases & Examples](#use-cases--examples)
9. [Deployment & Scalability](#deployment--scalability)
10. [Security Considerations](#security-considerations)
11. [References](#references)

---

## 1. Overview

The AI Agent Chat System is a conversational interface powered by Large Language Models (LLMs) and AI agents. It enables users to interact with intelligent agents that can provide suggestions, execute business workflows, and maintain context across multiple conversations.

### Key Objectives
- **User Experience**: Intuitive chat interface for seamless interactions
- **Scalability**: Handle multiple concurrent users and conversations
- **Intelligence**: Leverage LLMs for natural language understanding and generation
- **Performance**: Fast response times through intelligent caching
- **Persistence**: Reliable storage of conversation history

### Technology Stack
- **Frontend**: React/Vue.js (Web UI)
- **Backend**: Python FastAPI
- **LLM Framework**: LangChain/LangGraph
- **Database**: PostgreSQL
- **Cache**: Redis
- **LLM Provider**: OpenAI GPT-4, Anthropic Claude, or similar

---

## 2. System Architecture

### High-Level Architecture Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        A[Web Browser/UI]
    end
    
    subgraph "API Gateway"
        B[FastAPI Server]
    end
    
    subgraph "Service Layer"
        C[Conversation Service]
        D[Message Service]
        E[Agent Orchestrator]
    end
    
    subgraph "AI/LLM Layer"
        F[LangChain/LangGraph]
        G[LLM Provider<br/>GPT-4/Claude]
        H[Agent Tools]
    end
    
    subgraph "Data Layer"
        I[(PostgreSQL)]
        J[(Redis Cache)]
    end
    
    A -->|HTTP/WebSocket| B
    B --> C
    B --> D
    C --> I
    D --> I
    D --> E
    E --> F
    F --> G
    F --> H
    C --> J
    D --> J
    
    style A fill:#e1f5ff
    style B fill:#fff4e6
    style F fill:#f3e5f5
    style I fill:#e8f5e9
    style J fill:#ffebee
```

### Component Interaction Flow

```mermaid
sequenceDiagram
    participant U as User/UI
    participant API as FastAPI
    participant CS as Conversation Service
    participant MS as Message Service
    participant RC as Redis Cache
    participant AO as Agent Orchestrator
    participant LC as LangChain
    participant LLM as LLM Provider
    participant DB as PostgreSQL
    
    U->>API: Send Message
    API->>MS: Process Message
    MS->>RC: Check Cache
    alt Cache Hit
        RC-->>MS: Return Cached Response
    else Cache Miss
        MS->>DB: Save User Message
        MS->>AO: Route to Agent
        AO->>LC: Create Agent Chain
        LC->>LLM: Generate Response
        LLM-->>LC: Response
        LC-->>AO: Processed Response
        AO-->>MS: Agent Response
        MS->>DB: Save AI Response
        MS->>RC: Cache Response
    end
    MS-->>API: Return Response
    API-->>U: Display Response
```

---

## 3. Core Components

### 3.1 Web UI (Frontend)

The user interface provides an intuitive chat experience with features for conversation management.

**Key Features:**
- Conversation list sidebar
- Message input and display
- Real-time message streaming
- Conversation CRUD operations
- Markdown rendering for formatted responses

**UI Component Structure:**
```
src/
├── components/
│   ├── ConversationList/
│   │   ├── ConversationList.jsx
│   │   └── ConversationItem.jsx
│   ├── ChatWindow/
│   │   ├── ChatWindow.jsx
│   │   ├── MessageList.jsx
│   │   └── MessageInput.jsx
│   └── Header/
│       └── Header.jsx
├── services/
│   └── api.js
└── stores/
    └── chatStore.js
```

### 3.2 FastAPI Backend Service

The backend service handles HTTP requests, orchestrates business logic, and manages connections to data stores and LLM services.

**Project Structure:**
```
app/
├── main.py                 # FastAPI application entry
├── api/
│   ├── routes/
│   │   ├── conversations.py
│   │   ├── messages.py
│   │   └── agents.py
│   └── dependencies.py
├── services/
│   ├── conversation_service.py
│   ├── message_service.py
│   ├── cache_service.py
│   └── agent_service.py
├── models/
│   ├── conversation.py
│   ├── message.py
│   └── schemas.py
├── core/
│   ├── config.py
│   ├── database.py
│   └── redis.py
└── agents/
    ├── base_agent.py
    ├── customer_support_agent.py
    └── data_analysis_agent.py
```

### 3.3 Database Schema (PostgreSQL)

```mermaid
erDiagram
    USERS ||--o{ CONVERSATIONS : creates
    CONVERSATIONS ||--o{ MESSAGES : contains
    CONVERSATIONS ||--o{ CONVERSATION_METADATA : has
    
    USERS {
        uuid id PK
        string email
        string username
        timestamp created_at
        timestamp updated_at
    }
    
    CONVERSATIONS {
        uuid id PK
        uuid user_id FK
        string title
        string status
        timestamp created_at
        timestamp updated_at
    }
    
    MESSAGES {
        uuid id PK
        uuid conversation_id FK
        string role
        text content
        jsonb metadata
        timestamp created_at
    }
    
    CONVERSATION_METADATA {
        uuid id PK
        uuid conversation_id FK
        string key
        jsonb value
        timestamp created_at
    }
```

**SQL Schema Definition:**

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);

-- Conversation metadata table
CREATE TABLE conversation_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    key VARCHAR(100) NOT NULL,
    value JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(conversation_id, key)
);
```

---

## 4. Data Models

### 4.1 Pydantic Models (schemas.py)

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

class UserBase(BaseModel):
    email: str
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ConversationBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)

class ConversationCreate(ConversationBase):
    pass

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None

class Conversation(ConversationBase):
    id: UUID
    user_id: UUID
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class MessageBase(BaseModel):
    content: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None

class MessageCreate(MessageBase):
    conversation_id: UUID
    role: str = "user"

class Message(MessageBase):
    id: UUID
    conversation_id: UUID
    role: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[UUID] = None
    stream: bool = False

class ChatResponse(BaseModel):
    message_id: UUID
    content: str
    conversation_id: UUID
    created_at: datetime
```

### 4.2 SQLAlchemy ORM Models (models/conversation.py)

```python
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    title = Column(String(255), nullable=False)
    status = Column(String(50), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    metadata = relationship("ConversationMetadata", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

class ConversationMetadata(Base):
    __tablename__ = "conversation_metadata"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    key = Column(String(100), nullable=False)
    value = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="metadata")
```

---

## 5. API Design

### 5.1 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/conversations` | Create a new conversation |
| GET | `/api/v1/conversations` | List all conversations for a user |
| GET | `/api/v1/conversations/{id}` | Get a specific conversation with messages |
| PUT | `/api/v1/conversations/{id}` | Update conversation (e.g., title) |
| DELETE | `/api/v1/conversations/{id}` | Delete a conversation |
| POST | `/api/v1/messages` | Send a message to the chat system |
| GET | `/api/v1/messages/{conversation_id}` | Get messages for a conversation |
| POST | `/api/v1/chat/stream` | Stream chat responses (Server-Sent Events) |

### 5.2 Implementation Examples

**main.py - FastAPI Application Setup:**

```python
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import engine, Base
from app.core.redis import redis_client
from app.api.routes import conversations, messages

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    await redis_client.connect()
    yield
    # Shutdown
    await redis_client.close()

app = FastAPI(
    title="AI Agent Chat System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(conversations.router, prefix="/api/v1/conversations", tags=["conversations"])
app.include_router(messages.router, prefix="/api/v1/messages", tags=["messages"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**api/routes/conversations.py:**

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.core.database import get_db
from app.models.schemas import Conversation, ConversationCreate, ConversationUpdate
from app.services.conversation_service import ConversationService

router = APIRouter()

@router.post("/", response_model=Conversation, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation: ConversationCreate,
    user_id: UUID,  # In production, extract from JWT token
    db: Session = Depends(get_db)
):
    """Create a new conversation"""
    service = ConversationService(db)
    return await service.create_conversation(user_id, conversation)

@router.get("/", response_model=List[Conversation])
async def list_conversations(
    user_id: UUID,  # In production, extract from JWT token
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List all conversations for a user"""
    service = ConversationService(db)
    return await service.list_conversations(user_id, skip, limit)

@router.get("/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: UUID,
    user_id: UUID,  # In production, extract from JWT token
    db: Session = Depends(get_db)
):
    """Get a specific conversation"""
    service = ConversationService(db)
    conversation = await service.get_conversation(conversation_id, user_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@router.put("/{conversation_id}", response_model=Conversation)
async def update_conversation(
    conversation_id: UUID,
    conversation_update: ConversationUpdate,
    user_id: UUID,  # In production, extract from JWT token
    db: Session = Depends(get_db)
):
    """Update a conversation"""
    service = ConversationService(db)
    conversation = await service.update_conversation(conversation_id, user_id, conversation_update)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    user_id: UUID,  # In production, extract from JWT token
    db: Session = Depends(get_db)
):
    """Delete a conversation"""
    service = ConversationService(db)
    deleted = await service.delete_conversation(conversation_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return None
```

**api/routes/messages.py:**

```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.core.database import get_db
from app.models.schemas import Message, ChatRequest, ChatResponse
from app.services.message_service import MessageService

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def send_message(
    chat_request: ChatRequest,
    user_id: UUID,  # In production, extract from JWT token
    db: Session = Depends(get_db)
):
    """Send a message and get AI response"""
    service = MessageService(db)
    response = await service.process_message(user_id, chat_request)
    return response

@router.get("/{conversation_id}", response_model=List[Message])
async def get_messages(
    conversation_id: UUID,
    user_id: UUID,  # In production, extract from JWT token
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get messages for a conversation"""
    service = MessageService(db)
    messages = await service.get_messages(conversation_id, user_id, skip, limit)
    return messages

@router.post("/stream")
async def stream_message(
    chat_request: ChatRequest,
    user_id: UUID,
    db: Session = Depends(get_db)
):
    """Stream AI response for a message"""
    service = MessageService(db)
    
    async def generate():
        async for chunk in service.stream_message(user_id, chat_request):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 6. LLM Integration with LangChain

### 6.1 Agent Architecture

```mermaid
graph TB
    subgraph "Agent Orchestrator"
        A[User Input] --> B[Intent Classifier]
        B --> C{Route Decision}
    end
    
    subgraph "Specialized Agents"
        C -->|Customer Support| D[Support Agent]
        C -->|Data Analysis| E[Analytics Agent]
        C -->|General Chat| F[Conversational Agent]
        C -->|Task Execution| G[Workflow Agent]
    end
    
    subgraph "Tools & Resources"
        H[Database Tools]
        I[API Tools]
        J[Search Tools]
        K[Memory Store]
    end
    
    D --> H
    E --> H
    E --> I
    G --> I
    F --> K
    
    D --> L[Response Generator]
    E --> L
    F --> L
    G --> L
    
    L --> M[Final Response]
    
    style A fill:#e3f2fd
    style M fill:#c8e6c9
    style B fill:#fff9c4
```

### 6.2 LangChain Implementation

**agents/base_agent.py:**

```python
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from typing import List, Dict, Any
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=True
        )
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass
    
    @abstractmethod
    def get_tools(self) -> List:
        """Return the tools available to this agent"""
        pass
    
    def create_agent(self):
        """Create the agent executor"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        tools = self.get_tools()
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    async def run(self, user_input: str, conversation_history: List[Dict] = None) -> str:
        """Execute the agent with user input"""
        # Load conversation history into memory
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                if msg["role"] == "user":
                    self.memory.chat_memory.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    self.memory.chat_memory.add_ai_message(msg["content"])
        
        agent_executor = self.create_agent()
        response = await agent_executor.ainvoke({"input": user_input})
        return response["output"]
    
    async def stream(self, user_input: str, conversation_history: List[Dict] = None):
        """Stream the agent response"""
        if conversation_history:
            for msg in conversation_history[-10:]:
                if msg["role"] == "user":
                    self.memory.chat_memory.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    self.memory.chat_memory.add_ai_message(msg["content"])
        
        agent_executor = self.create_agent()
        
        async for chunk in agent_executor.astream({"input": user_input}):
            if "output" in chunk:
                yield chunk["output"]
```

**agents/customer_support_agent.py:**

```python
from langchain.tools import Tool
from langchain.agents import tool
from typing import List
from app.agents.base_agent import BaseAgent

class CustomerSupportAgent(BaseAgent):
    """Specialized agent for customer support queries"""
    
    def get_system_prompt(self) -> str:
        return """You are a helpful customer support agent. Your role is to:
        - Answer customer questions clearly and professionally
        - Look up order information when needed
        - Escalate complex issues to human agents
        - Maintain a friendly and empathetic tone
        - Provide accurate information from the knowledge base
        
        Always prioritize customer satisfaction and be honest if you don't know something.
        """
    
    def get_tools(self) -> List[Tool]:
        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the knowledge base for relevant information"""
            # Implementation would connect to actual KB
            return f"Knowledge base results for: {query}"
        
        @tool
        def lookup_order(order_id: str) -> str:
            """Look up order information by order ID"""
            # Implementation would query order database
            return f"Order {order_id}: Status - Shipped, ETA - 2 days"
        
        @tool
        def create_support_ticket(issue_description: str) -> str:
            """Create a support ticket for complex issues"""
            # Implementation would create actual ticket
            return f"Ticket created: #{12345}"
        
        return [
            search_knowledge_base,
            lookup_order,
            create_support_ticket
        ]
```

**agents/data_analysis_agent.py:**

```python
from langchain.tools import Tool
from langchain.agents import tool
from typing import List
import pandas as pd
from app.agents.base_agent import BaseAgent

class DataAnalysisAgent(BaseAgent):
    """Specialized agent for data analysis tasks"""
    
    def get_system_prompt(self) -> str:
        return """You are an expert data analyst. Your role is to:
        - Analyze datasets and provide insights
        - Create visualizations to illustrate findings
        - Answer questions about data patterns and trends
        - Write SQL queries to extract data
        - Explain statistical concepts in simple terms
        
        Always provide context and explain your methodology.
        """
    
    def get_tools(self) -> List[Tool]:
        @tool
        def execute_sql_query(query: str) -> str:
            """Execute a SQL query and return results"""
            # In production, use proper SQL execution with parameterization
            return "Query results: [sample data]"
        
        @tool
        def analyze_dataframe(description: str) -> str:
            """Analyze a pandas dataframe"""
            # Implementation would analyze actual data
            return "Analysis complete: Mean=45.2, Median=42, Std=12.3"
        
        @tool
        def create_visualization(chart_type: str, data_description: str) -> str:
            """Create a data visualization"""
            return f"Created {chart_type} chart showing {data_description}"
        
        return [
            execute_sql_query,
            analyze_dataframe,
            create_visualization
        ]
```

### 6.3 Agent Service Implementation

**services/agent_service.py:**

```python
from typing import Dict, List, Optional
from app.agents.base_agent import BaseAgent
from app.agents.customer_support_agent import CustomerSupportAgent
from app.agents.data_analysis_agent import DataAnalysisAgent
from langchain.chat_models import ChatOpenAI

class AgentService:
    """Service to orchestrate and route to appropriate agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {
            "customer_support": CustomerSupportAgent(),
            "data_analysis": DataAnalysisAgent(),
            "general": BaseAgent()  # Default conversational agent
        }
        self.classifier = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    async def classify_intent(self, user_message: str) -> str:
        """Classify user intent to route to appropriate agent"""
        prompt = f"""Classify the following user message into one of these categories:
        - customer_support: Questions about orders, products, returns, or support issues
        - data_analysis: Questions about data, statistics, analysis, or visualizations
        - general: General conversation, greetings, or other topics
        
        User message: "{user_message}"
        
        Respond with only the category name.
        """
        
        response = await self.classifier.ainvoke(prompt)
        intent = response.content.strip().lower()
        
        # Default to general if classification is unclear
        if intent not in self.agents:
            intent = "general"
        
        return intent
    
    async def process_message(
        self, 
        user_message: str, 
        conversation_history: List[Dict],
        agent_type: Optional[str] = None
    ) -> str:
        """Process a user message with the appropriate agent"""
        
        # Auto-classify if agent type not specified
        if not agent_type:
            agent_type = await self.classify_intent(user_message)
        
        agent = self.agents.get(agent_type, self.agents["general"])
        response = await agent.run(user_message, conversation_history)
        
        return response
    
    async def stream_message(
        self,
        user_message: str,
        conversation_history: List[Dict],
        agent_type: Optional[str] = None
    ):
        """Stream response from appropriate agent"""
        
        if not agent_type:
            agent_type = await self.classify_intent(user_message)
        
        agent = self.agents.get(agent_type, self.agents["general"])
        
        async for chunk in agent.stream(user_message, conversation_history):
            yield chunk
```

---

## 7. Caching Strategy

### 7.1 Redis Cache Architecture

```mermaid
graph LR
    A[User Request] --> B{Check Cache}
    B -->|Cache Hit| C[Return Cached Response]
    B -->|Cache Miss| D[Query LLM]
    D --> E[Store in Cache]
    E --> F[Return Response]
    
    G[Cache Invalidation] --> H{Trigger Type}
    H -->|Time-based| I[TTL Expiry]
    H -->|Event-based| J[Manual Invalidation]
    
    style C fill:#c8e6c9
    style D fill:#ffccbc
```

### 7.2 Cache Service Implementation

**core/redis.py:**

```python
from redis import asyncio as aioredis
from typing import Optional, Any
import json
from app.core.config import settings

class RedisClient:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Initialize Redis connection"""
        self.redis = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis:
            return None
        
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: int = 3600
    ) -> bool:
        """Set value in cache with expiration"""
        if not self.redis:
            return False
        
        serialized = json.dumps(value)
        return await self.redis.setex(key, expire, serialized)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis:
            return False
        
        return await self.redis.delete(key) > 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.redis:
            return False
        
        return await self.redis.exists(key) > 0
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.redis:
            return 0
        
        keys = await self.redis.keys(pattern)
        if keys:
            return await self.redis.delete(*keys)
        return 0

redis_client = RedisClient()
```

**services/cache_service.py:**

```python
from typing import Optional, List, Dict
from uuid import UUID
import hashlib
from app.core.redis import redis_client

class CacheService:
    """Service for managing cache operations"""
    
    CONVERSATION_TTL = 1800  # 30 minutes
    MESSAGE_LIST_TTL = 600   # 10 minutes
    LLM_RESPONSE_TTL = 3600  # 1 hour
    
    @staticmethod
    def _generate_cache_key(prefix: str, *args) -> str:
        """Generate a consistent cache key"""
        key_parts = [str(arg) for arg in args]
        return f"{prefix}:{':'.join(key_parts)}"
    
    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate hash for content-based caching"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get_cached_response(
        self, 
        user_message: str, 
        conversation_context: str
    ) -> Optional[str]:
        """Get cached LLM response for similar queries"""
        content_hash = self._hash_content(f"{conversation_context}:{user_message}")
        cache_key = self._generate_cache_key("llm_response", content_hash)
        
        cached = await redis_client.get(cache_key)
        return cached.get("response") if cached else None
    
    async def cache_llm_response(
        self,
        user_message: str,
        conversation_context: str,
        response: str
    ):
        """Cache LLM response for future similar queries"""
        content_hash = self._hash_content(f"{conversation_context}:{user_message}")
        cache_key = self._generate_cache_key("llm_response", content_hash)
        
        await redis_client.set(
            cache_key,
            {"response": response, "query": user_message},
            expire=self.LLM_RESPONSE_TTL
        )
```

---

## 8. Use Cases & Examples

### 8.1 Use Case 1: Customer Support Chatbot

**Scenario**: A customer wants to check their order status and asks about return policies.

```mermaid
sequenceDiagram
    participant C as Customer
    participant UI as Web UI
    participant API as FastAPI
    participant AS as Agent Service
    participant CS as Customer Support Agent
    participant DB as Database
    participant LLM as LLM Provider
    
    C->>UI: "What's the status of order #12345?"
    UI->>API: POST /api/v1/messages
    API->>AS: Process Message
    AS->>AS: Classify Intent → "customer_support"
    AS->>CS: Route to Customer Support Agent
    CS->>CS: Check available tools
    CS->>LLM: Generate response with tool use
    LLM-->>CS: "Use lookup_order tool"
    CS->>DB: Query order #12345
    DB-->>CS: Order status data
    CS->>LLM: Format response with order data
    LLM-->>CS: Natural language response
    CS-->>AS: Response
    AS-->>API: AI Response
    API->>DB: Save messages
    API-->>UI: Return response
    UI-->>C: Display order status
```

**Example Conversation:**

```
User: What's the status of order #12345?

Agent: Let me check that for you right away.

[Agent uses lookup_order tool]

Agent: Great news! Your order #12345 has been shipped and is currently in transit. 
      The estimated delivery date is November 18, 2025 (in 2 days). 
      You can track your package using tracking number: 1Z999AA10123456784.

User: What if I need to return it?

Agent: Of course! Our return policy allows you to return most items within 30 days 
      of delivery. Here's what you need to know:
      
      - Items must be unused and in original packaging
      - Free return shipping with the prepaid label we'll email you
      - Refund processed within 5-7 business days after we receive the item
      
      Would you like me to initiate a return for order #12345?
```

### 8.2 Use Case 2: Data Analysis Assistant

**Scenario**: A business analyst wants to analyze sales data and identify trends.

**Example Conversation:**

```
User: Can you analyze our Q3 sales data and show me which product categories 
      performed best?

Agent: I'll analyze the Q3 sales data for you. Let me query the database and 
       break down the performance by category.

[Agent uses execute_sql_query tool]

Agent: Here's what I found from Q3 2025 sales data:

Top Performing Categories:
1. Electronics: $2.4M revenue (32% of total) - Up 18% from Q2
2. Home & Garden: $1.8M revenue (24% of total) - Up 12% from Q2
3. Clothing: $1.5M revenue (20% of total) - Down 5% from Q2
4. Sports Equipment: $1.2M revenue (16% of total) - Up 25% from Q2
5. Books: $600K revenue (8% of total) - Stable

[Agent creates visualization]

Key Insights:
- Sports Equipment showed exceptional growth, likely due to seasonal demand
- Electronics continues to be the strongest category
- Clothing experienced a slight decline, possibly due to increased competition
- Overall Q3 revenue: $7.5M (up 14% year-over-year)

Would you like me to dive deeper into any specific category or create additional 
visualizations?
```

### 8.3 Use Case 3: Multi-Turn Conversation with Context

**Scenario**: A user engages in a complex conversation requiring context retention.

```mermaid
flowchart TD
    A[User: Plan a trip to Japan] --> B[Agent: Provides initial suggestions]
    B --> C[User: I prefer temples over cities]
    C --> D[Agent: Adjusts recommendations based on preference]
    D --> E[User: What's the best time to visit?]
    E --> F[Agent: Provides timing advice considering temple focus]
    F --> G[User: Book accommodations]
    G --> H[Agent: Searches with temple proximity in mind]
    
    style A fill:#e3f2fd
    style H fill:#c8e6c9
```

**Example Conversation:**

```
User: I'm planning a trip to Japan for 2 weeks. Can you help?

Agent: I'd love to help you plan your Japan trip! To give you the best 
       recommendations, could you tell me:
       - What are your main interests? (culture, food, nature, cities, etc.)
       - What's your budget range?
       - Any specific cities or regions you're interested in?

User: I love temples and traditional culture. Budget is moderate, around $3000 
      excluding flights.

Agent: Perfect! With your interest in temples and traditional culture, I recommend:

Week 1: Kyoto & Nara
- Kyoto has over 2,000 temples including Kinkaku-ji (Golden Pavilion) and 
  Fushimi Inari
- Day trip to Nara to see Todai-ji Temple and friendly deer
- Stay in a traditional ryokan for authentic experience

Week 2: Takayama & Tokyo
- Takayama offers well-preserved Edo-period streets
- Mount Koya for temple lodging experience
- Tokyo's Senso-ji and Meiji Shrine to contrast old and new

Budget Breakdown:
- Accommodations: $1,200 (mix of ryokan and hotels)
- JR Pass (14 days): $500
- Food: $700
- Attractions & Activities: $400
- Reserve: $200

Would you like me to create a detailed day-by-day itinerary?

User: Yes, and what's the best season to visit for temple photography?

Agent: Excellent question! For temple photography, I recommend:

Best Options:
1. Autumn (Late October - Early December) - HIGHLY RECOMMENDED
   - Stunning fall foliage frames temples beautifully
   - Clear skies and comfortable temperatures
   - Peak colors: mid-November

2. Spring (Late March - Early May)
   - Cherry blossoms create magical temple scenes
   - Peak bloom: late March to early April

Given your 2-week temple-focused itinerary, I'd suggest late October to early 
November. You'll capture:
- Red maple leaves at Kyoto temples
- Perfect lighting conditions
- Fewer crowds than cherry blossom season
- Comfortable walking weather for temple exploration

Should I adjust the itinerary with specific photography spots and golden hour 
timing for each temple?
```

### 8.4 Use Case 4: Business Workflow Automation

**Scenario**: An employee requests to automate expense report submission.

**Workflow Diagram:**

```mermaid
stateDiagram-v2
    [*] --> ReceiveRequest
    ReceiveRequest --> ValidateExpense
    ValidateExpense --> CheckPolicy: Valid
    ValidateExpense --> RequestCorrection: Invalid
    RequestCorrection --> ValidateExpense
    CheckPolicy --> CreateReport: Approved
    CheckPolicy --> EscalateManager: Exceeds Limit
    EscalateManager --> ManagerReview
    ManagerReview --> CreateReport: Approved
    ManagerReview --> Rejected: Denied
    CreateReport --> SubmitToFinance
    SubmitToFinance --> [*]
    Rejected --> [*]
```

**Example Conversation:**

```
User: I need to submit an expense report for my business trip last week.

Agent: I'll help you submit your expense report. Please provide the following:
       - Trip dates
       - Total amount
       - Expense categories (travel, meals, accommodation, etc.)
       - Receipt images or PDFs

User: Dates: Nov 10-12, 2025. Total: $850
      - Hotel: $300
      - Meals: $200
      - Transportation: $250
      - Conference fee: $100

Agent: Thank you! I'm processing your expense report for $850 (Nov 10-12, 2025).

[Agent validates against company policy]

Status: ✓ All items within policy limits
        ✓ Receipts required for items over $75
        
I've created Expense Report #ER-2025-1145. However, I need you to upload 
receipts for:
- Hotel ($300)
- Transportation ($250)

Would you like me to email you the submission link, or can you upload them now?

User: I'll upload them now. [Uploads receipts]

Agent: Perfect! Receipts received and attached to ER-2025-1145.

[Agent submits to finance system]

✓ Expense Report Successfully Submitted
- Report ID: ER-2025-1145
- Amount: $850
- Status: Under Review
- Expected reimbursement: 5-7 business days
- Finance team will review by: Nov 18, 2025

You'll receive an email notification once approved. Is there anything else you 
need help with?
```

---

## 9. Deployment & Scalability

### 9.1 Deployment Architecture

```mermaid
graph TB
    subgraph "Load Balancer Layer"
        LB[Load Balancer / Nginx]
    end
    
    subgraph "Application Layer - Multiple Instances"
        API1[FastAPI Instance 1]
        API2[FastAPI Instance 2]
        API3[FastAPI Instance 3]
    end
    
    subgraph "Cache Layer"
        REDIS1[Redis Master]
        REDIS2[Redis Replica]
    end
    
    subgraph "Database Layer"
        PG_MASTER[PostgreSQL Master]
        PG_REPLICA[PostgreSQL Replica]
    end
    
    subgraph "External Services"
        LLM[LLM API<br/>OpenAI/Anthropic]
    end
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> REDIS1
    API2 --> REDIS1
    API3 --> REDIS1
    
    REDIS1 --> REDIS2
    
    API1 --> PG_MASTER
    API2 --> PG_MASTER
    API3 --> PG_MASTER
    
    PG_MASTER --> PG_REPLICA
    
    API1 --> LLM
    API2 --> LLM
    API3 --> LLM
    
    style LB fill:#ffe0b2
    style REDIS1 fill:#ffccbc
    style PG_MASTER fill:#c8e6c9
```

### 9.2 Docker Compose Configuration

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  # FastAPI Application
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/chatdb
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./app:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 2G

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=chatdb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api

volumes:
  postgres_data:
  redis_data:
```

### 9.3 Kubernetes Deployment (Production)

**kubernetes/deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chat-api
  labels:
    app: chat-api
spec:
  replicas: 5
  selector:
    matchLabels:
      app: chat-api
  template:
    metadata:
      labels:
        app: chat-api
    spec:
      containers:
      - name: api
        image: your-registry/chat-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: chat-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: chat-secrets
              key: redis-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: chat-secrets
              key: openai-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: chat-api-service
spec:
  selector:
    app: chat-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chat-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: chat-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 9.4 Scalability Strategies

**Horizontal Scaling:**
- Multiple FastAPI instances behind load balancer
- Stateless application design
- Session data in Redis (shared across instances)

**Database Optimization:**
- Read replicas for query distribution
- Connection pooling (SQLAlchemy)
- Indexed queries on frequently accessed columns

**Caching Strategy:**
- Redis cluster for high availability
- Multi-level caching (L1: In-memory, L2: Redis)
- Cache warming for frequently accessed data

**Rate Limiting Implementation:**

```python
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    async def check_rate_limit(self, request: Request):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < 60
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        self.requests[client_ip].append(current_time)

rate_limiter = RateLimiter(requests_per_minute=100)

# Apply to routes
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    await rate_limiter.check_rate_limit(request)
    response = await call_next(request)
    return response
```

---

## 10. Security Considerations

### 10.1 Authentication & Authorization

**JWT Token Implementation:**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

SECRET_KEY = "your-secret-key"  # Use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return user_id
    except JWTError:
        raise credentials_exception

# Usage in routes
@router.post("/")
async def create_conversation(
    conversation: ConversationCreate,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    service = ConversationService(db)
    return await service.create_conversation(current_user, conversation)
```

### 10.2 Input Validation & Sanitization

**Preventing SQL Injection:**

```python
from sqlalchemy.orm import Session
from uuid import UUID

# GOOD: Using ORM parameters
def get_messages_safe(db: Session, conversation_id: UUID):
    return db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).all()

# BAD: String concatenation (DO NOT USE)
# def get_messages_unsafe(db: Session, conversation_id: str):
#     query = f"SELECT * FROM messages WHERE conversation_id = '{conversation_id}'"
#     return db.execute(query).fetchall()
```

**Content Sanitization:**

```python
import bleach
from typing import str

def sanitize_user_input(content: str) -> str:
    """Sanitize user input to prevent XSS attacks"""
    allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'code', 'pre']
    allowed_attrs = {}
    
    cleaned = bleach.clean(
        content,
        tags=allowed_tags,
        attributes=allowed_attrs,
        strip=True
    )
    return cleaned

# Usage in message processing
async def process_message(chat_request: ChatRequest):
    sanitized_message = sanitize_user_input(chat_request.message)
    # Process sanitized message
```

### 10.3 Prompt Injection Prevention

**LLM Security Best Practices:**

```python
class SecurePromptBuilder:
    """Build secure prompts to prevent injection attacks"""
    
    @staticmethod
    def build_system_prompt(agent_type: str) -> str:
        """Immutable system prompts"""
        prompts = {
            "customer_support": "You are a customer support agent...",
            "data_analysis": "You are a data analyst..."
        }
        return prompts.get(agent_type, "You are a helpful assistant.")
    
    @staticmethod
    def sanitize_user_input(user_message: str) -> str:
        """Remove potential injection attempts"""
        # Remove system prompt markers
        dangerous_patterns = [
            "ignore previous instructions",
            "system:",
            "assistant:",
            "[SYSTEM]",
            "<<SYS>>",
        ]
        
        cleaned = user_message
        for pattern in dangerous_patterns:
            cleaned = cleaned.replace(pattern, "")
        
        return cleaned.strip()
    
    @staticmethod
    def validate_tool_calls(tool_name: str, arguments: dict) -> bool:
        """Validate tool calls before execution"""
        allowed_tools = [
            "search_knowledge_base",
            "lookup_order",
            "execute_sql_query"
        ]
        
        if tool_name not in allowed_tools:
            return False
        
        # Additional argument validation
        # ...
        
        return True
```

### 10.4 Data Privacy & Compliance

**GDPR Compliance Features:**

```python
from datetime import datetime, timedelta

class DataPrivacyService:
    """Handle data privacy and compliance requirements"""
    
    async def anonymize_user_data(self, user_id: UUID):
        """Anonymize user data for GDPR compliance"""
        # Pseudonymize personal information
        # Replace with anonymized values
        pass
    
    async def export_user_data(self, user_id: UUID) -> dict:
        """Export all user data for GDPR data portability"""
        conversations = await self.get_user_conversations(user_id)
        messages = await self.get_user_messages(user_id)
        
        return {
            "user_id": str(user_id),
            "export_date": datetime.utcnow().isoformat(),
            "conversations": conversations,
            "messages": messages
        }
    
    async def delete_user_data(self, user_id: UUID):
        """Permanently delete user data (Right to be forgotten)"""
        # Delete all conversations and messages
        # Log deletion for audit trail
        pass
    
    async def data_retention_cleanup(self):
        """Automatically delete old data per retention policy"""
        retention_days = 365
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        # Delete conversations older than retention period
        # where user has not logged in
        pass
```

### 10.5 Security Checklist

- ✅ HTTPS/TLS encryption for all communications
- ✅ JWT-based authentication with token expiration
- ✅ Rate limiting to prevent abuse
- ✅ Input validation and sanitization
- ✅ SQL injection prevention (ORM parameterized queries)
- ✅ XSS prevention (content sanitization)
- ✅ Prompt injection protection
- ✅ Secrets management (environment variables, not hardcoded)
- ✅ Database encryption at rest
- ✅ Redis encryption in transit
- ✅ CORS policy configuration
- ✅ Regular security audits and penetration testing
- ✅ Logging and monitoring for suspicious activities
- ✅ GDPR compliance features

---

## 11. References

### Documentation
- **FastAPI**: https://fastapi.tiangolo.com/
- **LangChain**: https://python.langchain.com/docs/get_started/introduction
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **PostgreSQL**: https://www.postgresql.org/docs/
- **Redis**: https://redis.io/documentation
- **SQLAlchemy**: https://docs.sqlalchemy.org/

### LLM Providers
- **OpenAI API**: https://platform.openai.com/docs/
- **Anthropic Claude**: https://docs.anthropic.com/
- **Azure OpenAI**: https://learn.microsoft.com/en-us/azure/ai-services/openai/

### Best Practices & Guides
- **Prompt Engineering Guide**: https://www.promptingguide.ai/
- **LLM Security**: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- **API Security Best Practices**: https://owasp.org/www-project-api-security/
- **PostgreSQL Performance Tuning**: https://wiki.postgresql.org/wiki/Performance_Optimization
- **Redis Best Practices**: https://redis.io/docs/management/optimization/

### Tools & Libraries
- **Pydantic**: https://docs.pydantic.dev/ (Data validation)
- **Alembic**: https://alembic.sqlalchemy.org/ (Database migrations)
- **Docker**: https://docs.docker.com/ (Containerization)
- **Kubernetes**: https://kubernetes.io/docs/ (Orchestration)
- **Nginx**: https://nginx.org/en/docs/ (Load balancing)

### Monitoring & Observability
- **Prometheus**: https://prometheus.io/docs/ (Metrics)
- **Grafana**: https://grafana.com/docs/ (Visualization)
- **ELK Stack**: https://www.elastic.co/guide/ (Logging)
- **Sentry**: https://docs.sentry.io/ (Error tracking)

---

## Appendix: Quick Start Guide

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/ai-chat-system.git
cd ai-chat-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Start services with Docker Compose
docker-compose up -d postgres redis

# Run database migrations
alembic upgrade head

# Start FastAPI server
uvicorn app.main:app --reload

# API available at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

### Environment Variables (.env)

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/chatdb

# Redis
REDIS_URL=redis://localhost:6379

# LLM Provider
OPENAI_API_KEY=your_openai_api_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Security
SECRET_KEY=your_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application
ALLOWED_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
LOG_LEVEL=INFO
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_agents.py

# Run with verbose output
pytest -v
```

---

## Conclusion

This AI Agent Chat System design provides a robust, scalable foundation for building intelligent conversational applications. The architecture leverages modern technologies and best practices to deliver:

- **High Performance**: Through intelligent caching and optimization
- **Scalability**: Horizontal scaling and microservices architecture
- **Intelligence**: Advanced LLM integration with specialized agents
- **Security**: Comprehensive security measures and compliance features
- **Maintainability**: Clean code structure and comprehensive documentation

The system is designed to grow with your needs, supporting everything from simple chatbots to complex multi-agent workflows and enterprise integrations.