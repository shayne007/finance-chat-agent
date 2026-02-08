# C4 Dynamic Diagram - Chat Request Flow

## Overview
This diagram illustrates the dynamic flow of processing a chat request through the finance chat agent system.

```mermaid
C4Dynamic
  title Dynamic Diagram - Chat Request Processing Flow
  Person(user, "User", "End user of the chat system")

  Container(api, "FastAPI Application", "FastAPI", "Handles HTTP requests")
  ContainerDb(db, "SQLite Database", "SQLite", "Conversation and message storage")
  Container(task_queue, "Task Queue", "Redis", "Background task queue")

  Container_Boundary(agents, "AI Agents") {
    Component(router, "Finance Router", "LangGraph", "Routes queries")
    Component(github_agent, "GitHub Agent", "LangChain", "GitHub queries")
    Component(jira_agent, "JIRA Agent", "LangChain", "JIRA queries")
    Component(rag_agent, "RAG Agent", "LangChain", "Knowledge retrieval")
  }

  Container_Boundary(external, "External Services") {
    Component(openai, "OpenAI API", "REST API", "LLM processing")
    Component(github_api, "GitHub API", "REST API", "GitHub data")
    Component(jira_api, "JIRA API", "REST API", "JIRA data")
  }

  Rel(user, api, "1. POST /messages/chat-request", "JSON/HTTPS")
  Rel(api, db, "2. Save conversation entry")
  Rel(api, router, "3. Route query")
  Rel(router, db, "4. Update conversation context")

  Rel(router, github_agent, "5a. GitHub query", "If GitHub-related")
  Rel(router, jira_agent, "5b. JIRA query", "If JIRA-related")
  Rel(router, rag_agent, "5c. Knowledge query", "If knowledge needed")

  Rel(github_agent, github_api, "6. Query GitHub", "REST/HTTPS")
  Rel(jira_agent, jira_api, "6. Query JIRA", "REST/HTTPS")
  Rel(rag_agent, openai, "6. Process query", "REST/HTTPS")

  Rel(github_api, github_agent, "7. Return data")
  Rel(jira_api, jira_agent, "7. Return data")
  Rel(openai, rag_agent, "7. Return response")

  Rel(github_agent, router, "8. Return response")
  Rel(jira_agent, router, "8. Return response")
  Rel(rag_agent, router, "8. Return response")

  Rel(router, db, "9. Save agent response")
  Rel(db, api, "10. Return conversation ID")

  UpdateRelStyle(user, api, $textColor="blue", $offsetY="-30")
  UpdateRelStyle(api, router, $textColor="green", $offsetY="-30")
  UpdateRelStyle(router, github_agent, $textColor="purple", $offsetX="50")
  UpdateRelStyle(router, jira_agent, $textColor="orange", $offsetX="50")
  UpdateRelStyle(router, rag_agent, $textColor="red", $offsetX="50")
```

## Flow Description

### Phase 1: Request Reception
1. **User submits chat request** via HTTP POST to `/messages/chat-request`
2. **FastAPI saves conversation entry** to SQLite database
3. **Request routed to Finance Router** agent
4. **Router updates conversation context** with new message

### Phase 2: Query Routing
5a. **GitHub queries** → GitHub Agent
5b. **JIRA queries** → JIRA Agent
5c. **Knowledge queries** → RAG Agent

### Phase 3: External API Calls
6. **Agents query respective APIs**:
   - GitHub Agent queries GitHub API
   - JIRA Agent queries JIRA API
   - RAG Agent queries OpenAI API

### Phase 4: Response Aggregation
7. **API responses returned** to respective agents
8. **Agents process responses** and return to Finance Router
9. **Router saves agent response** to database
10. **System returns conversation ID** to user

## Key Features

### Asynchronous Processing
- Long-running queries can be processed in background
- Users can check status using conversation ID
- Task queue handles complex operations

### Error Handling
- Failed API calls are logged and retried
- Graceful degradation when external services unavailable
- Clear error messages returned to user

### State Management
- Conversation context maintained across turns
- Message history preserved for continuity
- Session state managed in database

## Performance Considerations

1. **Caching**: Frequently accessed data cached in Redis
2. **Batch Processing**: Multiple queries can be processed in parallel
3. **Background Tasks**: Repository documentation generation handled asynchronously
4. **Connection Pooling**: Efficient reuse of API connections
