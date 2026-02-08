# C4 Component Diagram - GitHub Integration

## Overview
This diagram shows the detailed components involved in GitHub integration and repository documentation generation.

```mermaid
C4Component
  title Component Diagram - GitHub Integration

  Container_Boundary(agents, "AI Agents") {
    Component(repo_doc_agent, "Repository Doc Agent", "LangChain", "Generates documentation from code")
    Component(github_agent, "GitHub Agent", "LangChain", "Handles GitHub queries")
  }

  Container_Boundary(github_services, "GitHub Services") {
    Component(github_client, "GitHub API Client", "PyGithub", "GitHub REST API wrapper")
    Component(repo_analyzer, "Repository Analyzer", "Python", "Code analysis engine")
    Component(doc_generator, "Documentation Generator", "Python", "Creates documentation from code")
    Component(diagram_generator, "Diagram Generator", "Python", "Creates architecture diagrams")
    Component(knowledge_base, "Knowledge Base", "Vector DB", "Stores documentation fragments")
  }

  Container_Boundary(tasks, "Background Tasks") {
    Component(repo_processor, "Repository Processor", "Celery", "Processes repository data")
    Component(doc_builder, "Documentation Builder", "Celery", "Assembles documentation")
    Component(indexer, "Indexer", "Celery", "Indexes documentation for search")
  }

  Container_Boundary(data, "Data Storage") {
    ComponentDb(db, "SQLite Database", "SQLite", "Repository metadata and documentation")
    Component(vector_db, "Vector Database", "FAISS/Chroma", "Document embeddings")
    Component(cache, "Redis Cache", "Redis", "Repository cache")
  }

  Rel(repo_doc_agent, github_client, "Uses GitHub API")
  Rel(github_client, repo_analyzer, "Fetches repository data")
  Rel(repo_analyzer, doc_generator, "Provides analyzed code")
  Rel(doc_generator, diagram_generator, "Creates diagrams")
  Rel(doc_generator, knowledge_base, "Stores fragments")

  Rel(repo_doc_agent, repo_processor, "Queues processing tasks")
  Rel(repo_processor, doc_builder, "Assembles documentation")
  Rel(doc_builder, indexer, "Indexes documents")

  Rel(doc_builder, db, "Stores generated docs")
  Rel(indexer, vector_db, "Stores embeddings")
  Rel(cache, github_client, "Caches repository data")

  Rel(repo_doc_agent, knowledge_base, "Queries for context")
  Rel(github_agent, github_client, "Uses for GitHub queries")
```

## Component Details

### 1. Repository Doc Agent
- **Purpose**: Generates comprehensive documentation from code repositories
- **Capabilities**:
  - Analyzes repository structure
  - Generates documentation for code
  - Creates architecture diagrams
  - Builds searchable knowledge base

### 2. GitHub Agent
- **Purpose**: Handles GitHub-specific queries
- **Capabilities**:
  - Query repositories, issues, PRs
  - Analyze commit history
  - Get contributor information
  - Manage GitHub workflows

### 3. GitHub API Client
- **Technology**: PyGithub
- **Purpose**: Wrapper for GitHub REST API
- **Features**:
  - Repository operations
  - Issue/PR management
  - GitHub GraphQL support
  - Rate limiting handling

### 4. Repository Analyzer
- **Purpose**: Analyzes code structure and content
- **Features**:
  - Parses source code
  - Identifies classes and functions
  - Extracts documentation strings
  - Builds dependency graphs

### 5. Documentation Generator
- **Purpose**: Creates human-readable documentation
- **Features**:
  - Generates API documentation
  - Creates usage examples
  - Explains code patterns
  - Generates best practices

### 6. Diagram Generator
- **Purpose**: Creates architecture diagrams
- **Features**:
  - Component diagrams
  - Sequence diagrams
  - Class diagrams
  - Package structure visualization

### 7. Background Tasks
- **Repository Processor**: Handles large repository processing
- **Documentation Builder**: Assembles final documentation
- **Indexer**: Creates searchable embeddings

### 8. Data Storage
- **SQLite Database**: Stores metadata and documentation
- **Vector Database**: For semantic search capabilities
- **Redis Cache**: For frequently accessed repository data

## Processing Flow

1. **Repository Analysis**:
   - Fetch repository via GitHub API
   - Analyze code structure
   - Extract documentation
   - Generate diagrams

2. **Documentation Generation**:
   - Create comprehensive docs
   - Include code examples
   - Add best practices
   - Build knowledge base

3. **Storage and Indexing**:
   - Store in SQLite database
   - Create embeddings for search
   - Cache frequently accessed data
