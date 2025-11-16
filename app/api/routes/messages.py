from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.core.database import get_db
from app.models.schemas import Message as MessageSchema, ChatRequest, ChatResponse
from app.services.message_service import MessageService


router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def send_message(chat_request: ChatRequest, user_id: UUID, db: Session = Depends(get_db)):
    service = MessageService(db)
    return await service.process_message(user_id, chat_request)


@router.get("/{conversation_id}", response_model=List[MessageSchema])
async def get_messages(conversation_id: UUID, user_id: UUID, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    service = MessageService(db)
    return await service.get_messages(conversation_id, user_id, skip, limit)