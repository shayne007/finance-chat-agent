# Jira Issue Content Knowledge Base - System Design Document

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Design Principles](#design-principles)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Implementation Details](#implementation-details)
7. [Use Cases](#use-cases)
8. [Deployment and Scalability](#deployment-and-scalability)
9. [Future Enhancements](#future-enhancements)

---

## Executive Summary

This document outlines the design of a **Jira Issue Content Knowledge Base** - an intelligent backend service that ingests, processes, and enables semantic search over Jira issues using LLM embeddings. The system leverages PostgreSQL with PGVector extension, OpenAI's embedding models, and LangChain for building a robust RAG (Retrieval-Augmented Generation) pipeline suitable for financial QA applications.

### Key Capabilities
- Automated Jira issue ingestion with content deduplication
- Vector embeddings generation using OpenAI Ada model
- Semantic similarity search using PGVector
- RAG-based AI agent for intelligent Q&A
- Extensible architecture for multiple data sources

---

## System Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        A[Jira API]
        B[GitHub Code]
        C[Log Files]
        D[DDL SQL]
    end
    
    subgraph "Ingestion Layer"
        E[Data Fetcher]
        F[Content Parser]
        G[Deduplication Engine]
    end
    
    subgraph "Processing Layer"
        H[Text Chunker]
        I[OpenAI Embedding API]
        J[Metadata Extractor]
    end
    
    subgraph "Storage Layer"
        K[(PostgreSQL + PGVector)]
        L[(Source Content Store)]
        M[(MD5 Hash Index)]
    end
    
    subgraph "Query Layer"
        N[Similarity Search]
        O[RAG Agent]
        P[Response Generator]
    end
    
    A --> E
    B -.-> E
    C -.-> E
    D -.-> E
    
    E --> F
    F --> G
    G --> H
    G --> L
    G --> M
    
    H --> I
    I --> K
    J --> K
    
    N --> K
    N --> O
    O --> P
    
    style A fill:#e1f5ff
    style K fill:#ffe1e1
    style O fill:#e1ffe1
```

### Architecture Overview

The system follows a **layered architecture** with clear separation of concerns:

1. **Data Sources Layer**: External systems providing raw content
2. **Ingestion Layer**: Fetches and preprocesses data
3. **Processing Layer**: Transforms content into embeddings
4. **Storage Layer**: Persistent storage with vector search capabilities
5. **Query Layer**: Handles search and RAG operations

---

## Design Principles

### 1. Single Responsibility Principle
Each component has a well-defined, focused responsibility ensuring maintainability and testability.

### 2. Idempotency
All ingestion operations are idempotent through MD5-based deduplication, allowing safe re-runs without data corruption.

### 3. Extensibility
Plugin-based architecture allows easy addition of new data sources without modifying core logic.

### 4. Scalability
Designed for horizontal scaling with stateless components and efficient batch processing.

### 5. Data Integrity
Source content is preserved in original form alongside embeddings for audit and re-processing capabilities.

### 6. Performance Optimization
Implements chunking strategies, batch embedding generation, and efficient vector indexing.

---

## Core Components

### Component Diagram

```mermaid
graph LR
    subgraph "Core Services"
        A[IngestionService]
        B[EmbeddingService]
        C[VectorStoreService]
        D[RAGService]
    end
    
    subgraph "Data Adapters"
        E[JiraAdapter]
        F[GitHubAdapter]
        G[LogAdapter]
    end
    
    subgraph "Utilities"
        H[ChunkingEngine]
        I[DeduplicationManager]
        J[MetadataExtractor]
    end
    
    E --> A
    F -.-> A
    G -.-> A
    
    A --> I
    A --> H
    A --> B
    B --> C
    
    C --> D
    
    H -.-> J
```

---

## Data Flow

### Ingestion Pipeline

```mermaid
sequenceDiagram
    participant Jira as Jira API
    participant Ing as Ingestion Service
    participant Dedup as Deduplication Manager
    participant Chunk as Chunking Engine
    participant Embed as Embedding Service
    participant DB as PostgreSQL/PGVector
    
    Ing->>Jira: Fetch Issues (JQL Query)
    Jira-->>Ing: Return Issue List
    
    loop For Each Issue
        Ing->>Dedup: Check MD5 Hash
        Dedup->>DB: Query Hash Index
        
        alt Content Already Exists
            DB-->>Dedup: Hash Found
            Dedup-->>Ing: Skip (No Changes)
        else New or Updated Content
            DB-->>Dedup: Hash Not Found
            Dedup-->>Ing: Proceed
            
            Ing->>DB: Store Original Content
            Ing->>Chunk: Split into Chunks
            Chunk-->>Ing: Return Chunks
            
            loop For Each Chunk
                Ing->>Embed: Generate Embedding
                Embed->>OpenAI: Call /embeddings API
                OpenAI-->>Embed: Return Vector
                Embed-->>Ing: Return Embedding
                Ing->>DB: Store Vector + Metadata
            end
            
            Ing->>DB: Update MD5 Index
        end
    end
```

### Query Pipeline

```mermaid
sequenceDiagram
    participant User
    participant RAG as RAG Service
    participant Embed as Embedding Service
    participant Vector as Vector Store
    participant LLM as OpenAI LLM
    
    User->>RAG: Submit Question
    RAG->>Embed: Generate Query Embedding
    Embed->>OpenAI: Call /embeddings API
    OpenAI-->>Embed: Return Vector
    Embed-->>RAG: Return Query Vector
    
    RAG->>Vector: Similarity Search
    Vector->>Vector: Cosine Similarity (PGVector)
    Vector-->>RAG: Return Top-K Documents
    
    RAG->>RAG: Build Context
    RAG->>LLM: Generate Answer with Context
    LLM-->>RAG: Return Response
    RAG-->>User: Return Answer + Sources
```

---

## Implementation Details

### 1. Database Schema Setup

```python
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import hashlib
from datetime import datetime

Base = declarative_base()

class SourceDocument(Base):
    """Stores original source content with metadata"""
    __tablename__ = 'source_documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_type = Column(String(50), nullable=False)  # 'jira', 'github', 'log', etc.
    source_id = Column(String(255), nullable=False, unique=True)  # e.g., 'JIRA-1234'
    content_md5 = Column(String(32), nullable=False, index=True)
    raw_content = Column(Text, nullable=False)
    metadata_json = Column(Text)  # JSON string of additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SourceDocument(source_id='{self.source_id}', type='{self.source_type}')>"


class DatabaseConfig:
    """Database configuration and initialization"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
    
    def initialize_schema(self):
        """Create all tables and enable PGVector extension"""
        Base.metadata.create_all(self.engine)
        
        # Enable PGVector extension
        with self.engine.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
    
    def get_vector_store(self, collection_name: str = "jira_embeddings"):
        """Initialize PGVector store for embeddings"""
        return PGVector(
            connection_string=self.connection_string,
            collection_name=collection_name,
            embedding_function=None,  # Will be set by EmbeddingService
        )
```

### 2. Jira Data Adapter

```python
from jira import JIRA
from typing import List, Dict, Any
import json

class JiraAdapter:
    """Adapter for fetching data from Jira API"""
    
    def __init__(self, server: str, username: str, api_token: str):
        self.client = JIRA(
            server=server,
            basic_auth=(username, api_token)
        )
    
    def fetch_issues(self, jql: str, max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch issues using JQL query
        
        Args:
            jql: Jira Query Language string
            max_results: Maximum number of results to fetch
            
        Returns:
            List of issue dictionaries with normalized structure
        """
        issues = self.client.search_issues(
            jql_str=jql,
            maxResults=max_results,
            expand='changelog,comments'
        )
        
        return [self._normalize_issue(issue) for issue in issues]
    
    def _normalize_issue(self, issue) -> Dict[str, Any]:
        """Convert Jira issue object to normalized dictionary"""
        fields = issue.fields
        
        # Extract comments
        comments = []
        if hasattr(fields, 'comment') and fields.comment:
            comments = [
                {
                    'author': c.author.displayName,
                    'body': c.body,
                    'created': c.created
                }
                for c in fields.comment.comments
            ]
        
        # Build comprehensive content string
        content_parts = [
            f"Issue Key: {issue.key}",
            f"Summary: {fields.summary}",
            f"Description: {fields.description or 'No description'}",
            f"Status: {fields.status.name}",
            f"Priority: {fields.priority.name if fields.priority else 'None'}",
            f"Reporter: {fields.reporter.displayName if fields.reporter else 'Unknown'}",
            f"Assignee: {fields.assignee.displayName if fields.assignee else 'Unassigned'}",
        ]
        
        if comments:
            content_parts.append("\nComments:")
            for idx, comment in enumerate(comments, 1):
                content_parts.append(
                    f"{idx}. [{comment['author']}]: {comment['body']}"
                )
        
        full_content = "\n".join(content_parts)
        
        return {
            'source_id': issue.key,
            'source_type': 'jira',
            'content': full_content,
            'metadata': {
                'issue_key': issue.key,
                'summary': fields.summary,
                'status': fields.status.name,
                'priority': fields.priority.name if fields.priority else None,
                'created': fields.created,
                'updated': fields.updated,
                'project': fields.project.key,
                'issue_type': fields.issuetype.name,
                'labels': fields.labels if hasattr(fields, 'labels') else [],
            }
        }
```

### 3. Deduplication Manager

```python
import hashlib
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

class DeduplicationManager:
    """Manages content deduplication using MD5 hashing"""
    
    def __init__(self, db_session: Session):
        self.session = db_session
    
    def compute_md5(self, content: str) -> str:
        """Compute MD5 hash of content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def check_duplicate(self, source_id: str, content: str) -> Optional[SourceDocument]:
        """
        Check if content already exists or has been updated
        
        Returns:
            - None if content is new
            - SourceDocument if content exists and hasn't changed
            - SourceDocument with different MD5 if content was updated
        """
        content_md5 = self.compute_md5(content)
        
        existing_doc = self.session.query(SourceDocument).filter_by(
            source_id=source_id
        ).first()
        
        if not existing_doc:
            return None  # New document
        
        if existing_doc.content_md5 == content_md5:
            return existing_doc  # Unchanged
        
        # Content updated - return for update processing
        return existing_doc
    
    def store_source(self, source_id: str, source_type: str, 
                     content: str, metadata: Dict[str, Any]) -> SourceDocument:
        """Store or update source document"""
        content_md5 = self.compute_md5(content)
        
        existing_doc = self.session.query(SourceDocument).filter_by(
            source_id=source_id
        ).first()
        
        if existing_doc:
            # Update existing document
            existing_doc.content_md5 = content_md5
            existing_doc.raw_content = content
            existing_doc.metadata_json = json.dumps(metadata)
            existing_doc.updated_at = datetime.utcnow()
            doc = existing_doc
        else:
            # Create new document
            doc = SourceDocument(
                source_id=source_id,
                source_type=source_type,
                content_md5=content_md5,
                raw_content=content,
                metadata_json=json.dumps(metadata)
            )
            self.session.add(doc)
        
        self.session.commit()
        return doc
```

### 4. Text Chunking Engine

```python
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChunkingEngine:
    """Handles intelligent text chunking for embedding generation"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split document into chunks with metadata preservation
        
        Args:
            content: Full document content
            metadata: Document metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = self.splitter.split_text(content)
        
        chunk_documents = []
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk_text)
            })
            
            chunk_documents.append({
                'content': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunk_documents
```

### 5. Embedding Service

```python
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any
import os

class EmbeddingService:
    """Handles embedding generation using OpenAI API"""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-ada-002"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=model
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embeddings.embed_query(text)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        return self.embeddings.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """Return dimension of embedding vectors"""
        # Ada-002 produces 1536-dimensional vectors
        return 1536 if self.model == "text-embedding-ada-002" else 1536
```

### 6. Ingestion Service

```python
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from langchain_postgres import PGVector

class IngestionService:
    """Orchestrates the complete ingestion pipeline"""
    
    def __init__(
        self,
        db_session: Session,
        vector_store: PGVector,
        embedding_service: EmbeddingService,
        chunking_engine: ChunkingEngine,
        dedup_manager: DeduplicationManager
    ):
        self.db_session = db_session
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.chunking_engine = chunking_engine
        self.dedup_manager = dedup_manager
    
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Ingest a batch of documents
        
        Args:
            documents: List of document dictionaries with 'source_id', 
                      'source_type', 'content', and 'metadata'
        
        Returns:
            Statistics dictionary with counts
        """
        stats = {
            'processed': 0,
            'new': 0,
            'updated': 0,
            'skipped': 0,
            'chunks_created': 0
        }
        
        for doc in documents:
            stats['processed'] += 1
            
            # Check for duplicates
            existing = self.dedup_manager.check_duplicate(
                doc['source_id'], 
                doc['content']
            )
            
            if existing and existing.content_md5 == self.dedup_manager.compute_md5(doc['content']):
                stats['skipped'] += 1
                continue
            
            # Store original document
            if existing:
                stats['updated'] += 1
                # Delete old embeddings for this source
                self._delete_old_embeddings(doc['source_id'])
            else:
                stats['new'] += 1
            
            self.dedup_manager.store_source(
                source_id=doc['source_id'],
                source_type=doc['source_type'],
                content=doc['content'],
                metadata=doc['metadata']
            )
            
            # Chunk the document
            chunks = self.chunking_engine.chunk_document(
                content=doc['content'],
                metadata=doc['metadata']
            )
            
            # Generate embeddings and store
            self._store_chunks(chunks, doc['source_id'])
            stats['chunks_created'] += len(chunks)
        
        return stats
    
    def _delete_old_embeddings(self, source_id: str):
        """Delete existing embeddings for a source document"""
        # This depends on PGVector's delete functionality
        # Implementation varies based on LangChain version
        pass
    
    def _store_chunks(self, chunks: List[Dict[str, Any]], source_id: str):
        """Generate embeddings and store chunks in vector store"""
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add source_id to all metadata
        for metadata in metadatas:
            metadata['source_id'] = source_id
        
        # Store in vector database
        self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas
        )
```

### 7. RAG Service

```python
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGService:
    """Retrieval-Augmented Generation service for Q&A"""
    
    def __init__(
        self,
        vector_store: PGVector,
        embedding_service: EmbeddingService,
        llm_model: str = "gpt-4",
        api_key: str = None
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0
        )
        
        # Configure retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Custom prompt template
        self.prompt_template = PromptTemplate(
            template="""You are a helpful AI assistant with access to a knowledge base of Jira issues and related documents. 
Use the following context to answer the question. If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Answer: Provide a clear, concise answer based on the context above. Include relevant issue keys and details when applicable.""",
            input_variables=["context", "question"]
        )
        
        # Build QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
    
    def query(self, question: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute RAG query
        
        Args:
            question: User question
            filters: Optional metadata filters for retrieval
            
        Returns:
            Dictionary with answer and source documents
        """
        # Apply filters if provided
        if filters:
            self.retriever.search_kwargs['filter'] = filters
        
        result = self.qa_chain.invoke({"query": question})
        
        return {
            'answer': result['result'],
            'source_documents': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in result['source_documents']
            ]
        }
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform pure similarity search without LLM
        
        Args:
            query: Search query
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of matching documents with scores
        """
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filters
        )
        
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score)
            }
            for doc, score in results
        ]
```

### 8. Main Application Controller

```python
import os
from typing import Optional

class JiraKnowledgeBase:
    """Main application controller"""
    
    def __init__(
        self,
        db_connection_string: str,
        jira_server: str,
        jira_username: str,
        jira_api_token: str,
        openai_api_key: str
    ):
        # Initialize database
        self.db_config = DatabaseConfig(db_connection_string)
        self.db_config.initialize_schema()
        
        # Initialize services
        self.embedding_service = EmbeddingService(api_key=openai_api_key)
        self.chunking_engine = ChunkingEngine()
        
        # Get database session
        self.session = self.db_config.Session()
        self.dedup_manager = DeduplicationManager(self.session)
        
        # Initialize vector store
        self.vector_store = PGVector(
            connection_string=db_connection_string,
            collection_name="jira_embeddings",
            embedding_function=self.embedding_service.embeddings
        )
        
        # Initialize ingestion service
        self.ingestion_service = IngestionService(
            db_session=self.session,
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            chunking_engine=self.chunking_engine,
            dedup_manager=self.dedup_manager
        )
        
        # Initialize Jira adapter
        self.jira_adapter = JiraAdapter(
            server=jira_server,
            username=jira_username,
            api_token=jira_api_token
        )
        
        # Initialize RAG service
        self.rag_service = RAGService(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            api_key=openai_api_key
        )
    
    def ingest_jira_issues(self, jql: str, max_results: int = 1000) -> Dict[str, int]:
        """
        Ingest Jira issues based on JQL query
        
        Args:
            jql: Jira Query Language string
            max_results: Maximum number of issues to fetch
            
        Returns:
            Statistics dictionary
        """
        print(f"Fetching Jira issues with JQL: {jql}")
        issues = self.jira_adapter.fetch_issues(jql, max_results)
        
        print(f"Fetched {len(issues)} issues. Starting ingestion...")
        stats = self.ingestion_service.ingest_documents(issues)
        
        print(f"Ingestion complete: {stats}")
        return stats
    
    def ask_question(self, question: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ask a question using RAG
        
        Args:
            question: User question
            filters: Optional metadata filters
            
        Returns:
            Answer with source documents
        """
        return self.rag_service.query(question, filters)
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of similar documents
        """
        return self.rag_service.similarity_search(query, k)
    
    def close(self):
        """Clean up resources"""
        self.session.close()


# Example usage
if __name__ == "__main__":
    kb = JiraKnowledgeBase(
        db_connection_string="postgresql://user:pass@localhost:5432/jira_kb",
        jira_server="https://your-company.atlassian.net",
        jira_username="your-email@company.com",
        jira_api_token="your-api-token",
        openai_api_key="sk-..."
    )
    
    # Ingest issues
    stats = kb.ingest_jira_issues(
        jql="project = FIN AND created >= -30d"
    )
    
    # Ask questions
    result = kb.ask_question(
        "What are the critical bugs in the payment processing module?"
    )
    print(f"Answer: {result['answer']}")
    
    kb.close()
```

---

## Use Cases

### Use Case 1: Financial Bug Tracking Analysis

**Scenario**: A financial services company needs to analyze recurring payment processing bugs.

**Implementation**:
```python
kb = JiraKnowledgeBase(...)

# Ingest all payment-related issues
kb.ingest_jira_issues(
    jql="project = PAYMENTS AND component = 'Processing' AND type = Bug"
)

# Query for patterns
result = kb.ask_question(
    "What are the common causes of payment timeout errors in the last quarter?",
    filters={'metadata.project': 'PAYMENTS'}
)

print(result['answer'])
# Output: "Based on 23 related issues, the primary causes are:
# 1. Database connection pool exhaustion (PAYMENTS-1234, PAYMENTS-1456)
# 2. Third-party API rate limiting (PAYMENTS-1789)
# 3. Network latency during peak hours (PAYMENTS-2001)..."
```

### Use Case 2: Compliance Audit Trail

**Scenario**: Retrieve all security-related changes for regulatory compliance.

**Implementation**:
```python
# Search for security-related discussions
results = kb.search_similar(
    query="PCI DSS compliance data encryption requirements",
    k=10
)

# Filter by date range
filtered_results = [
    r for r in results 
    if r['metadata'].get('created', '') >= '2024-01-01'
]

# Generate compliance report
for doc in filtered_results:
    print(f"Issue: {doc['metadata']['issue_key']}")
    print(f"Summary: {doc['metadata']['summary']}")
    print(f"Similarity: {doc['similarity_score']:.2f}")
```

### Use Case 3: Automated Triage Support

**Scenario**: Help support team triage new issues by finding similar past issues.

**Implementation**:
```python
new_issue_description = """
Customer reports that wire transfer initiated on 2025-01-15 
has not appeared in beneficiary account after 48 hours. 
Transaction ID: WTX-2025-001234. No error message shown.
"""

# Find similar historical issues
similar_issues = kb.search_similar(
    query=new_issue_description,
    k=5
)

print("Similar past issues:")
for issue in similar_issues:
    print(f"\n{issue['metadata']['issue_key']}: {issue['metadata']['summary']}")
    print(f"Resolution: Check comments in original issue")
    print(f"Similarity: {issue['similarity_score']:.2f}")
```

### Use Case 4: Knowledge Gap Identification

**Scenario**: Identify areas lacking documentation.

**Implementation**:
```python
# Query various topics
topics = [
    "API authentication flow",
    "Database failover procedure",
    "Payment reconciliation process",
    "Fraud detection algorithms"
]

gaps = []
for topic in topics:
    results = kb.search_similar(topic, k=1)
    if not results or results[0]['similarity_score'] < 0.7:
        gaps.append(topic)

print("Documentation gaps found:")
for gap in gaps:
    print(f"- {gap}")
```

---

##