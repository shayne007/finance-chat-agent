from sqlalchemy.orm import Session
from uuid import UUID
from typing import List
from datetime import datetime
import uuid as uuidlib

from app.models.conversation import Conversation, Message
from app.models.schemas import ChatRequest, ChatResponse, Message as MessageSchema
from app.services.conversation_service import ConversationService
from app.models.schemas import ConversationCreate
from app.agents.finance_agent import FinanceAgent


class MessageService:
    def __init__(self, db: Session):
        self.db = db
        self.finance_agent = FinanceAgent()

    async def process_message(self, user_id: UUID, chat_request: ChatRequest) -> ChatResponse:
        conversation_id = chat_request.conversation_id
        if not conversation_id:
            conv_service = ConversationService(self.db)
            conv = await conv_service.create_conversation(user_id, ConversationCreate(title="New Conversation"))
            conversation_id = conv.id

        conv = (
            self.db.query(Conversation)
            .filter(Conversation.id == str(conversation_id), Conversation.user_id == str(user_id))
            .first()
        )
        if not conv:
            raise ValueError("Conversation not found or not owned by user")

        user_msg = Message(conversation_id=str(conversation_id), role="user", content=chat_request.message)
        self.db.add(user_msg)
        self.db.commit()
        self.db.refresh(user_msg)

        history = [
            {"role": m.role, "content": m.content}
            for m in (
                self.db.query(Message)
                .filter(Message.conversation_id == str(conversation_id))
                .order_by(Message.created_at.asc())
                .all()
            )
        ]

        reply = await self.finance_agent.run(chat_request.message, history)

        ai_msg = Message(conversation_id=str(conversation_id), role="assistant", content=reply)
        self.db.add(ai_msg)
        self.db.commit()
        self.db.refresh(ai_msg)

        return ChatResponse(message_id=ai_msg.id, content=ai_msg.content, conversation_id=ai_msg.conversation_id, created_at=ai_msg.created_at)

    async def get_messages(self, conversation_id: UUID, user_id: UUID, skip: int, limit: int) -> List[MessageSchema]:
        conv = (
            self.db.query(Conversation)
            .filter(Conversation.id == str(conversation_id), Conversation.user_id == str(user_id))
            .first()
        )
        if not conv:
            return []
        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == str(conversation_id))
            .order_by(Message.created_at.asc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return messages