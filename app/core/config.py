import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./chat.db")
    ALLOWED_ORIGINS: list[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ])
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    CELERY_TASK_ALWAYS_EAGER: bool = os.getenv("CELERY_TASK_ALWAYS_EAGER", "1") == "1"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    REDIS_CHECKPOINT_URL: str = os.getenv("REDIS_CHECKPOINT_URL", "")
    JIRA_DOMAIN: str = os.getenv("JIRA_DOMAIN", "")
    JIRA_EMAIL: str = os.getenv("JIRA_EMAIL", "")
    JIRA_API_TOKEN: str = os.getenv("JIRA_API_TOKEN", "")
    JIRA_PROJECT_KEY: str = os.getenv("JIRA_PROJECT_KEY", "PROJ")

    # RAG Agent Settings
    RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    RAG_LLM_MODEL: str = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    RAG_SEARCH_K: int = int(os.getenv("RAG_SEARCH_K", "5"))
    # Optional: For PGVector support (PostgreSQL vector store)
    RAG_VECTOR_DB_URL: str = os.getenv("RAG_VECTOR_DB_URL", "")
    RAG_VECTOR_COLLECTION: str = os.getenv("RAG_VECTOR_COLLECTION", "kb_embeddings")


settings = Settings()
