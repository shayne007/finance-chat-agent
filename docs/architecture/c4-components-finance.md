# C4 Component Diagram - Finance Agent System

## Overview
This diagram shows the internal components of the AI agent system and how they interact with each other.

```mermaid
C4Component
  title Component Diagram - Finance Agent System

  Container(api, "FastAPI Application", "FastAPI", "REST API endpoints")

  Container_Boundary(agents, "AI Agents Container") {
    Component(finance_router, "Finance Router Agent", "LangGraph", "Routes queries to appropriate sub-agents")
    Component(github_agent, "GitHub Agent", "LangChain", "Handles GitHub-related queries")
    Component(jira_agent, "JIRA Agent", "LangChain", "Handles project management queries")
    Component(rag_agent, "RAG Agent", "LangChain", "Knowledge retrieval and augmentation")
    Component(repo_doc_agent, "Repository Doc Agent", "LangChain", "Generates documentation from code")
  }

  Container_Boundary(services, "Business Logic Services") {
    Component(conv_service, "Conversation Service", "Python", "Manages conversation state")
    Component(message_service, "Message Service", "Python", "Processes messages")
    Component(repo_service, "Repository Service", "Python", "Manages repository operations")
  }

  Container_Boundary(clients, "External API Clients") {
    Component(github_client, "GitHub API Client", "Python", "GitHub API wrapper")
    Component(jira_client, "JIRA API Client", "Python", "JIRA API wrapper")
  }

  Container_Boundary(core, "Core Infrastructure") {
    Component(config, "Configuration", "Python", "Application settings")
    Component(db, "Database Models", "SQLAlchemy", "Data models")
    Component(celery, "Celery Integration", "Celery", "Background task management")
  }

  Rel(api, finance_router, "Routes incoming queries")
  Rel(finance_router, github_agent, "Routes GitHub queries")
  Rel(finance_router, jira_agent, "Routes JIRA queries")
  Rel(finance_router, rag_agent, "Routes knowledge queries")
  Rel(finance_router, repo_doc_agent, "Routes documentation requests")

  Rel(github_agent, github_client, "Uses API client")
  Rel(jira_agent, jira_client, "Uses API client")

  Rel(conv_service, db, "Reads/writes conversation data")
  Rel(message_service, db, "Reads/writes message data")
  Rel(repo_service, db, "Manages repository data")

  Rel(repo_doc_agent, celery, "Queues async tasks")
  Rel(celery, repo_service, "Executes repository operations")

  Rel(finance_router, conv_service, "Updates conversation state")
  Rel(conv_service, message_service, "Coordinates message flow")

  Rel(config, api, "Provides configuration")
  Rel(config, finance_router, "Provides agent configuration")
  Rel(config, github_agent, "Provides agent configuration")
  Rel(config, jira_agent, "Provides agent configuration")
  Rel(config, rag_agent, "Provides agent configuration")
  Rel(config, repo_doc_agent, "Provides agent configuration")
  Rel(config, db, "Provides core settings")
  Rel(config, celery, "Provides core settings")
```

## Component Details

### 1. Finance Router Agent
- **Purpose**: Entry point for all finance-related queries
- **Responsibilities**:
  - Analyzes user intent
  - Routes queries to appropriate sub-agents
  - Maintains conversation context
  - Aggregates responses

### 2. Specialized Agents
- **GitHub Agent**: Handles queries about repositories, issues, pull requests, and GitHub operations
- **JIRA Agent**: Manages queries about projects, issues, boards, and project tracking
- **RAG Agent**: Provides knowledge retrieval from documentation and databases
- **Repository Documentation Agent**: Generates documentation from code repositories

### 3. Business Logic Services
- **Conversation Service**: Manages conversation state, turns, and context
- **Message Service**: Processes incoming and outgoing messages
- **Repository Service**: Handles repository-related operations and data

### 4. External API Clients
- **GitHub API Client**: Wraps GitHub REST API for repository operations
- **JIRA API Client**: Wraps JIRA REST API for project management operations

### 5. Core Infrastructure
- **Configuration**: Manages application settings and environment variables
- **Database Models**: Defines data structures for conversations, messages, and users
- **Celery Integration**: Handles background task processing and scheduling

## Key Data Flows

1. **Query Flow**:
   - User query → FastAPI → Finance Router → Specialized Agent
   - Agent processes query using appropriate tools/clients
   - Response flows back through the same path

2. **Background Processing**:
   - Repository documentation generation queued via Celery
   - Async data fetching from GitHub/JIRA APIs
   - Results stored in database

3. **State Management**:
   - Conversation service maintains context across turns
   - Message service tracks message history
   - Repository service manages repository metadata
