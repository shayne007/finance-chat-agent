import os
import hashlib
from typing import List, Dict, Any, Optional, Sequence
from datetime import datetime, timezone
from dataclasses import dataclass, field

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate


@dataclass
class SearchResult:
    """Represents a semantic search result"""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "similarity_score": self.similarity_score
        }


@dataclass
class IngestionStats:
    """Statistics from document ingestion"""
    processed: int = 0
    new: int = 0
    updated: int = 0
    skipped: int = 0
    chunks_created: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processed": self.processed,
            "new": self.new,
            "updated": self.updated,
            "skipped": self.skipped,
            "chunks_created": self.chunks_created,
            "errors": self.errors
        }


class DeduplicationManager:
    """Manages content deduplication using MD5 hashing"""

    def __init__(self):
        self._hash_index: Dict[str, Dict[str, str]] = {}

    def compute_md5(self, content: str) -> str:
        """Compute MD5 hash of content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def check_duplicate(self, source_id: str, content: str) -> Optional[str]:
        """
        Check if content already exists or has been updated

        Returns:
            - None if content is new
            - existing_md5 if content exists and hasn't changed
            - different_md5 if content was updated (returns old MD5)
        """
        content_md5 = self.compute_md5(content)

        if source_id not in self._hash_index:
            return None  # New document

        existing_md5 = self._hash_index[source_id].get("content_md5")
        if existing_md5 == content_md5:
            return existing_md5  # Unchanged

        return existing_md5  # Updated (returns old MD5)

    def store_source(self, source_id: str, content: str, metadata: Dict[str, Any]) -> str:
        """Store or update source document hash"""
        content_md5 = self.compute_md5(content)

        self._hash_index[source_id] = {
            "content_md5": content_md5,
            "metadata": metadata,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        return content_md5

    def remove_source(self, source_id: str) -> bool:
        """Remove a source from the hash index"""
        return self._hash_index.pop(source_id, None) is not None

    def get_all_sources(self) -> List[str]:
        """Get all registered source IDs"""
        return list(self._hash_index.keys())


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

    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split document into chunks with metadata preservation

        Args:
            content: Full document content
            metadata: Document metadata to attach to each chunk

        Returns:
            List of Document objects with content and metadata
        """
        chunks = self.splitter.split_text(content)

        documents = []
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk_text),
                'chunked_at': datetime.now(timezone.utc).isoformat()
            })

            documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return documents


class VectorStoreManager:
    """Manages vector store backend for embeddings"""

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        # Default to in-memory vector store
        self.vector_store = InMemoryVectorStore(embeddings)
        self._store_type = "memory"

    def add_documents(self, documents: Sequence[Document]) -> List[str]:
        """Add documents to the vector store"""
        return self.vector_store.add_documents(documents)

    def delete_documents(self, ids: List[str]) -> Optional[bool]:
        """Delete documents from the vector store"""
        if hasattr(self.vector_store, 'delete'):
            return self.vector_store.delete(ids)
        return None

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Search for similar documents with scores"""
        return self.vector_store.similarity_search_with_score(query, k=k, filter=filter)

    def as_retriever(self, **kwargs):
        """Get a retriever interface"""
        return self.vector_store.as_retriever(**kwargs)


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent for semantic search and Q&A.

    This agent provides:
    - Semantic search in knowledge base using vector embeddings
    - Document ingestion with automatic chunking
    - RAG-based question answering
    - Content deduplication
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=embedding_model
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            openai_api_key=self.api_key,
            temperature=0
        )

        # Initialize components
        self.dedup_manager = DeduplicationManager()
        self.chunking_engine = ChunkingEngine(chunk_size, chunk_overlap)
        self.vector_store_manager = VectorStoreManager(self.embeddings)

        # RAG prompt template
        self.rag_prompt = PromptTemplate(
            template="""You are a helpful AI assistant with access to a knowledge base.
Use the following context to answer the question. If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Answer: Provide a clear, concise answer based on the context above.""",
            input_variables=["context", "question"]
        )

    def ingest_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> IngestionStats:
        """
        Ingest a batch of documents into the knowledge base.

        Args:
            documents: List of document dictionaries with 'source_id',
                      'content', and optional 'metadata'

        Returns:
            IngestionStats with counts and any errors
        """
        stats = IngestionStats()

        for doc in documents:
            stats.processed += 1

            source_id = doc.get('source_id')
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})

            if not source_id or not content:
                stats.errors.append(f"Invalid document: {doc}")
                continue

            try:
                # Check for duplicates
                existing_md5 = self.dedup_manager.check_duplicate(source_id, content)

                if existing_md5 == self.dedup_manager.compute_md5(content):
                    stats.skipped += 1
                    continue

                # Store/update source hash
                self.dedup_manager.store_source(source_id, content, metadata)

                if existing_md5:
                    stats.updated += 1
                    # Note: In a real vector store with persistent backend,
                    # we would delete old embeddings here
                else:
                    stats.new += 1

                # Chunk the document
                chunks = self.chunking_engine.chunk_document(content, metadata)

                # Store in vector store
                self.vector_store_manager.add_documents(chunks)
                stats.chunks_created += len(chunks)

            except Exception as e:
                stats.errors.append(f"Error processing {source_id}: {str(e)}")

        return stats

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform semantic similarity search.

        Args:
            query: Search query text
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        results = self.vector_store_manager.similarity_search_with_score(
            query=query,
            k=k,
            filter=filters
        )

        return [
            SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                similarity_score=float(score)
            )
            for doc, score in results
        ]

    def ask_question(
        self,
        question: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ask a question using RAG (Retrieval-Augmented Generation).

        Args:
            question: User question
            k: Number of context documents to retrieve
            filters: Optional metadata filters for retrieval

        Returns:
            Dictionary with answer and source documents
        """
        # Retrieve relevant documents
        search_results = self.similarity_search(question, k=k, filters=filters)

        if not search_results:
            return {
                "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                "source_documents": [],
                "confidence": 0.0
            }

        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results, 1):
            source_info = result.metadata.get('source_id', 'Unknown')
            context_parts.append(f"[Source {i}: {source_info}]\n{result.content}")

        context = "\n\n".join(context_parts)

        # Generate answer using LLM
        prompt = self.rag_prompt.format(context=context, question=question)
        response = self.llm.invoke([HumanMessage(prompt)])

        # Calculate confidence based on similarity scores
        avg_score = sum(r.similarity_score for r in search_results) / len(search_results)
        confidence = max(0.0, min(1.0, avg_score))

        return {
            "answer": response.content,
            "source_documents": [r.to_dict() for r in search_results],
            "confidence": confidence,
            "context_used": len(search_results)
        }

    def search_similar_documents(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar documents with optional minimum score threshold.

        Args:
            query: Search query text
            k: Maximum number of results to return
            filters: Optional metadata filters
            min_score: Minimum similarity score (0-1)

        Returns:
            List of result dictionaries
        """
        results = self.similarity_search(query, k=k, filters=filters)

        # Filter by minimum score
        filtered_results = [r for r in results if r.similarity_score >= min_score]

        return [r.to_dict() for r in filtered_results]

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        sources = self.dedup_manager.get_all_sources()

        return {
            "total_sources": len(sources),
            "source_ids": sources,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "chunk_size": self.chunking_engine.chunk_size,
            "chunk_overlap": self.chunking_engine.chunk_overlap
        }

    def clear_knowledge_base(self) -> Dict[str, Any]:
        """Clear all documents from the knowledge base"""
        sources_count = len(self.dedup_manager.get_all_sources())
        self.dedup_manager = DeduplicationManager()
        self.vector_store_manager = VectorStoreManager(self.embeddings)

        return {
            "cleared": True,
            "previous_sources_count": sources_count
        }

    def remove_document(self, source_id: str) -> bool:
        """Remove a document from the knowledge base"""
        return self.dedup_manager.remove_source(source_id)

    @property
    def chunk_size(self) -> int:
        return self.chunking_engine.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self.chunking_engine.chunk_overlap
