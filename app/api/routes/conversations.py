from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.core.database import get_db
from app.models.schemas import Conversation as ConversationSchema, ConversationCreate, ConversationUpdate
from app.services.conversation_service import ConversationService


router = APIRouter()


@router.post("/", response_model=ConversationSchema, status_code=status.HTTP_201_CREATED)
async def create_conversation(conversation: ConversationCreate, user_id: UUID, db: Session = Depends(get_db)):
    service = ConversationService(db)
    return await service.create_conversation(user_id, conversation)


@router.get("/", response_model=List[ConversationSchema])
async def list_conversations(user_id: UUID, skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    service = ConversationService(db)
    return await service.list_conversations(user_id, skip, limit)


@router.get("/{conversation_id}", response_model=ConversationSchema)
async def get_conversation(conversation_id: UUID, user_id: UUID, db: Session = Depends(get_db)):
    service = ConversationService(db)
    conversation = await service.get_conversation(conversation_id, user_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.put("/{conversation_id}", response_model=ConversationSchema)
async def update_conversation(conversation_id: UUID, conversation_update: ConversationUpdate, user_id: UUID, db: Session = Depends(get_db)):
    service = ConversationService(db)
    conversation = await service.update_conversation(conversation_id, user_id, conversation_update)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: UUID, user_id: UUID, db: Session = Depends(get_db)):
    service = ConversationService(db)
    deleted = await service.delete_conversation(conversation_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return None