# C4 System Context - Finance Chat Agent

## Overview
The system context diagram shows the finance chat agent as a central system interacting with various users and external services.

```mermaid
C4Context
title System Context - Finance Chat Agent

Person(user, "Finance Professional", "Uses chat interface for finance queries")
Person(developer, "Developer", "Manages repositories and documentation")
Person(pm, "Project Manager", "Tracks project issues and status")

System(finance_agent, "Finance Chat Agent", "AI-powered chat system for finance")
System_Ext(github, "GitHub", "Code hosting and collaboration platform")
System_Ext(jira, "JIRA", "Project management and issue tracking")
System_Ext(openai, "OpenAI", "LLM services for AI capabilities")
System_Ext(redis, "Redis", "Message broker and caching")
System_Ext(sqlite, "SQLite", "Persistent data storage")

Rel(user, finance_agent, "Submits finance queries")
Rel(developer, finance_agent, "Generates repository documentation")
Rel(pm, finance_agent, "Tracks project issues via JIRA")

Rel(finance_agent, github, "Queries repositories/issues/PRs")
Rel(finance_agent, jira, "Queries issues/boards/projects")
Rel(finance_agent, openai, "Processes natural language queries")
Rel(finance_agent, redis, "Queues async tasks")
Rel(finance_agent, sqlite, "Stores conversations/messages")
```

## Key Interactions

1. **Primary Users**:
   - Finance professionals use the chat interface for finance-related queries
   - Developers use the system to generate documentation from repositories
   - Project managers use it to track project issues and status

2. **External Dependencies**:
   - OpenAI provides the core AI capabilities through LLM models
   - GitHub integration allows querying repositories, issues, and pull requests
   - JIRA integration provides project management data
   - Redis serves as the message broker for asynchronous processing
   - SQLite stores all conversation and user data

3. **Core Purpose**:
   - The finance chat agent acts as a central hub that connects users with multiple domain-specific AI agents
   - It routes queries to the appropriate agent (finance, GitHub, JIRA, or documentation)
   - Manages the conversation flow and maintains context across interactions
