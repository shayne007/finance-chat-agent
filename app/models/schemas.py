from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID


class ConversationBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class ConversationCreate(ConversationBase):
    pass


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None


class Conversation(ConversationBase):
    id: UUID
    user_id: UUID
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageBase(BaseModel):
    content: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class MessageCreate(MessageBase):
    conversation_id: UUID
    role: str = "user"


class Message(MessageBase):
    id: UUID
    conversation_id: UUID
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[UUID] = None
    stream: bool = False


class ChatResponse(BaseModel):
    message_id: UUID
    content: str
    conversation_id: UUID
    created_at: datetime


class ChatQueuedResponse(BaseModel):
    message_id: UUID
    task_id: str
    status: str
    response_message_id: Optional[UUID] = None
    content: Optional[str] = None
    conversation_id: Optional[UUID] = None
    created_at: Optional[datetime] = None


class AgentRequest(BaseModel):
    user_input: str
    context: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    output: str
    ticket_id: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None