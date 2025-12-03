from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.core.database import get_db
from app.models.schemas import Message as MessageSchema, ChatRequest, ChatResponse, ChatQueuedResponse
from app.services.message_service import MessageService
from app.models.conversation import Message, Conversation
from app.core.config import settings
import json


router = APIRouter()


@router.post("/chat-request", response_model=ChatQueuedResponse)
async def send_message(chat_request: ChatRequest, user_id: UUID, db: Session = Depends(get_db)):
    service = MessageService(db)
    return await service.queue_message(user_id, chat_request)


@router.get("/chat-request/{message_id}", response_model=ChatQueuedResponse)
async def get_chat_status(message_id: UUID, user_id: UUID, db: Session = Depends(get_db)):
    msg = db.query(Message).filter(Message.id == str(message_id)).first()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    conv = db.query(Conversation).filter(Conversation.id == msg.conversation_id, Conversation.user_id == str(user_id)).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Message not found")
    meta = {}
    if msg.meta:
        try:
            meta = json.loads(msg.meta)
        except Exception:
            meta = {}
    task_id = meta.get("task_id")
    if meta.get("status") == "completed":
        response_id = meta.get("response_message_id")
        content = None
        conv_id = None
        created_at = None
        if response_id:
            ai_msg = db.query(Message).filter(Message.id == response_id).first()
            if ai_msg:
                content = ai_msg.content
                conv_id = ai_msg.conversation_id
                created_at = ai_msg.created_at
        return ChatQueuedResponse(
            message_id=message_id,
            task_id=task_id or "",
            status="completed",
            response_message_id=response_id,
            content=content,
            conversation_id=conv_id,
            created_at=created_at,
        )
    status = meta.get("status") or "unknown"
    return ChatQueuedResponse(message_id=message_id, task_id=task_id or "", status=status)