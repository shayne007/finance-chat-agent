import json
from app.core.celery_app import celery_app
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.conversation import Conversation, Message
from app.agents.finance_agent import FinanceAgent
import asyncio


@celery_app.task
def process_message_task(message_id: str, user_id: str) -> str:
    db: Session = SessionLocal()
    try:
        user_msg = db.query(Message).filter(Message.id == message_id).first()
        if not user_msg:
            return "not_found"

        conv = db.query(Conversation).filter(Conversation.id == user_msg.conversation_id, Conversation.user_id == user_id).first()
        if not conv:
            return "unauthorized"

        history = [
            {"role": m.role, "content": m.content}
            for m in db.query(Message).filter(Message.conversation_id == conv.id).order_by(Message.created_at.asc()).all()
        ]

        agent = FinanceAgent()
        reply = asyncio.run(agent.run(user_msg.content, history))

        ai_msg = Message(conversation_id=conv.id, role="assistant", content=reply, meta=json.dumps({"parent_message_id": message_id}))
        db.add(ai_msg)
        db.commit()
        db.refresh(ai_msg)

        meta = {}
        try:
            meta = json.loads(user_msg.meta) if user_msg.meta else {}
        except Exception:
            meta = {}
        meta.update({"status": "completed", "response_message_id": ai_msg.id})
        user_msg.meta = json.dumps(meta)
        db.add(user_msg)
        db.commit()
        return ai_msg.id
    finally:
        db.close()