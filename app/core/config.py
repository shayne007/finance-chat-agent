import os
from dataclasses import dataclass, field
from typing import Optional, Union

from pydantic import BaseModel, field_validator, Field


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


class GitHubSettings(BaseModel):
    """Settings for GitHub Agent integration.

    This class loads configuration from environment variables with Pydantic validation.
    """

    # Required fields
    token: str = Field(default="")

    # Optional fields with defaults
    default_repo: Optional[str] = Field(default=None)
    base_url: str = Field(default="https://api.github.com")
    timeout: int = Field(default=30)
    max_retries: int = Field(default=3)
    enabled: bool = Field(default=False)

    @field_validator("token")
    @classmethod
    def validate_token(cls, v: str) -> str:
        """Validate that the token is not empty."""
        if not v or not v.strip():
            raise ValueError("GITHUB_TOKEN cannot be empty")
        return v

    @field_validator("enabled", mode="before")
    @classmethod
    def parse_enabled(cls, v: Union[str, bool, None]) -> bool:
        """Parse enabled field from string to boolean."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v) if v is not None else False

    @field_validator("timeout", "max_retries", mode="before")
    @classmethod
    def parse_int(cls, v: Union[str, int, None]) -> int:
        """Parse integer fields from string to int."""
        if isinstance(v, int):
            return v
        try:
            return int(v)
        except (ValueError, TypeError):
            raise ValueError(f"Must be a valid integer, got: {v}")


settings = Settings()


def load_github_settings() -> GitHubSettings:
    """Load GitHub settings from environment variables.

    Returns:
        GitHubSettings instance populated from environment variables.
    """
    field_map = {
        "token": "GITHUB_TOKEN",
        "default_repo": "GITHUB_DEFAULT_REPO",
        "base_url": "GITHUB_BASE_URL",
        "timeout": "GITHUB_TIMEOUT",
        "max_retries": "GITHUB_MAX_RETRIES",
        "enabled": "GITHUB_AGENT_ENABLED",
    }

    kwargs = {}
    for field_name, env_name in field_map.items():
        env_value = os.getenv(env_name)
        if env_value is not None:
            kwargs[field_name] = env_value

    return GitHubSettings(**kwargs)


github_settings = load_github_settings()


class MCPSettings(BaseModel):
    """Settings for MCP Server integration.

    This class loads configuration for the Model Context Protocol server.
    """

    # Required fields
    github_server_enabled: bool = Field(default=False, alias="ENABLED")

    # Optional fields with defaults
    github_server_port: int = Field(default=8081, alias="PORT")
    github_max_tools: int = Field(default=50, alias="MAX_TOOLS")

    @field_validator("github_server_enabled", mode="before")
    @classmethod
    def parse_enabled(cls, v: Union[str, bool, None]) -> bool:
        """Parse enabled field from string to boolean."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v) if v is not None else False

    @field_validator("github_server_port", mode="before")
    @classmethod
    def parse_port(cls, v: Union[str, int, None]) -> int:
        """Parse port field from string to int with validation."""
        if isinstance(v, int):
            port = v
        elif isinstance(v, str):
            try:
                port = int(v)
            except ValueError:
                raise ValueError(f"Must be a valid integer, got: {v}")
        else:
            port = 8081

        if port < 1 or port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got: {port}")
        return port

    @field_validator("github_max_tools", mode="before")
    @classmethod
    def parse_max_tools(cls, v: Union[str, int, None]) -> int:
        """Parse max_tools field from string to int."""
        if isinstance(v, int):
            return v
        try:
            return int(v)
        except (ValueError, TypeError):
            raise ValueError(f"Must be a valid integer, got: {v}")


mcp_settings = MCPSettings()

