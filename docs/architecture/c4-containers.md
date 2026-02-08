# C4 Container Diagram - Finance Chat Agent

## Overview
The container diagram shows the main applications, databases, and services that make up the finance chat agent system.

```mermaid
C4Container
  title Container Diagram - Finance Chat Agent

  Person(user, "User", "Finance professionals, developers, PMs")
  Person(developer, "Developer", "Code repository management")

  Container_Boundary(api_app, "FastAPI Web Application") {
    Container(web_app, "FastAPI App", "FastAPI, Python", "Main web server and API endpoints")
    Container(agents, "AI Agent System", "LangChain, LangGraph", "Multi-agent coordination")
    Container(mcp_server, "MCP Server", "Model Context Protocol", "GitHub integration server")
  }

  Container_Boundary(background_tasks, "Background Processing") {
    Container(celery_worker, "Celery Worker", "Celery, Python", "Async task processing")
    Container(task_queue, "Task Queue", "Redis", "Message broker for tasks")
  }

  Container_Boundary(data_storage, "Data Persistence") {
    ContainerDb(db, "SQLite Database", "SQLite", "Conversation history and user data")
    Container(cache, "Redis Cache", "Redis", "Session and temporary data")
  }

  Container_Ext(openai, "OpenAI API", "REST API", "LLM services")
  Container_Ext(github_api, "GitHub API", "REST API", "Repository management")
  Container_Ext(jira_api, "JIRA API", "REST API", "Issue tracking")

  Rel(user, web_app, "Submits chat requests via HTTP")
  Rel(web_app, agents, "Routes queries to appropriate agent")
  Rel(agents, openai, "Sends natural language queries")
  Rel(web_app, db, "Saves conversation history")
  Rel(web_app, cache, "Caches session data")

  Rel(web_app, task_queue, "Queues async tasks")
  Rel(task_queue, celery_worker, "Processes background tasks")
  Rel(celery_worker, github_api, "Fetches repository data")
  Rel(celery_worker, jira_api, "Fetches issue data")
  Rel(celery_worker, db, "Updates task results")

  Rel(mcp_server, github_api, "GitHub API integration")
  Rel(agents, mcp_server, "Uses MCP tools")

  Rel(developer, web_app, "Generates repository documentation")
  Rel(celery_worker, db, "Stores generated docs")
```

## Container Details

### 1. FastAPI Web Application
- **Technology**: FastAPI, Python 3.9+
- **Purpose**: Main API server handling HTTP requests
- **Key Features**:
  - RESTful API endpoints for chat requests
  - Authentication and authorization
  - Request validation and response formatting
  - Conversation management

### 2. AI Agent System
- **Technology**: LangChain, LangGraph, OpenAI
- **Purpose**: Multi-agent coordination and query routing
- **Key Agents**:
  - Finance Agent: Handles finance-related queries
  - GitHub Agent: Manages GitHub interactions
  - JIRA Agent: Handles project management queries
  - RAG Agent: Provides knowledge retrieval
  - Repository Documentation Agent: Generates code documentation

### 3. MCP Server
- **Technology**: Model Context Protocol
- **Purpose**: Enhanced GitHub integration
- **Features**: Tool-based AI capabilities for GitHub operations

### 4. Celery Worker
- **Technology**: Celery, Redis
- **Purpose**: Background task processing
- **Tasks**:
  - Repository analysis and documentation generation
  - GitHub/JIRA data fetching
  - Async message processing

### 5. SQLite Database
- **Technology**: SQLite, SQLAlchemy
- **Purpose**: Persistent data storage
- **Schema**:
  - Conversations table
  - Messages table
  - User data
  - Generated documentation

### 6. Redis Cache
- **Technology**: Redis
- **Purpose**: Session management and caching
- **Uses**:
  - Session storage
  - Rate limiting
  - Temporary data caching