from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import Base, engine
from app.core.config import settings
from app.api.routes.conversations import router as conversations_router
from app.api.routes.messages import router as messages_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="AI Agent Chat System", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversations_router, prefix="/api/v1/conversations", tags=["conversations"])
app.include_router(messages_router, prefix="/api/v1/messages", tags=["messages"])


@app.get("/health")
async def health():
    return {"status": "healthy"}