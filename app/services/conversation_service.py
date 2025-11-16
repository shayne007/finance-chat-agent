from sqlalchemy.orm import Session
from uuid import UUID
from typing import List, Optional
from datetime import datetime
import uuid as uuidlib

from app.models.conversation import Conversation
from app.models.schemas import ConversationCreate, ConversationUpdate


class ConversationService:
    def __init__(self, db: Session):
        self.db = db

    async def create_conversation(self, user_id: UUID, conversation: ConversationCreate) -> Conversation:
        conv = Conversation(user_id=str(user_id), title=conversation.title, status="active")
        self.db.add(conv)
        self.db.commit()
        self.db.refresh(conv)
        return conv

    async def list_conversations(self, user_id: UUID, skip: int, limit: int) -> List[Conversation]:
        return (
            self.db.query(Conversation)
            .filter(Conversation.user_id == str(user_id))
            .order_by(Conversation.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    async def get_conversation(self, conversation_id: UUID, user_id: UUID) -> Optional[Conversation]:
        return (
            self.db.query(Conversation)
            .filter(Conversation.id == str(conversation_id), Conversation.user_id == str(user_id))
            .first()
        )

    async def update_conversation(self, conversation_id: UUID, user_id: UUID, update: ConversationUpdate) -> Optional[Conversation]:
        conv = await self.get_conversation(conversation_id, user_id)
        if not conv:
            return None
        if update.title is not None:
            conv.title = update.title
        if update.status is not None:
            conv.status = update.status
        conv.updated_at = datetime.utcnow()
        self.db.add(conv)
        self.db.commit()
        self.db.refresh(conv)
        return conv

    async def delete_conversation(self, conversation_id: UUID, user_id: UUID) -> bool:
        conv = await self.get_conversation(conversation_id, user_id)
        if not conv:
            return False
        self.db.delete(conv)
        self.db.commit()
        return True