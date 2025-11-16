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


settings = Settings()