import json
import os
import logging
from langchain_openai import ChatOpenAI
from app.agents.github_agent import GitHubAgent
from app.clients.github_client import GitHubClient
from app.mcp.github_server import GitHubMCPServer
from app.core.config import settings, github_settings
from app.core.celery_app import celery_app
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.conversation import Conversation, Message
from app.agents.finance_agent import FinanceAgent


logger = logging.getLogger(__name__)


@celery_app.task
def process_message_task(message_id: str, user_id: str) -> str:
    db: Session = SessionLocal()
    try:
        user_msg = db.query(Message).filter(Message.id == message_id).first()
        if not user_msg:
            return "not_found"

        conv = db.query(Conversation).filter(Conversation.id == user_msg.conversation_id, Conversation.user_id == user_id).first()
        if not conv:
            meta = {}
            try:
                meta = json.loads(user_msg.meta) if user_msg.meta else {}
            except Exception:
                meta = {}
            meta.update({"status": "failed", "error": "unauthorized"})
            user_msg.meta = json.dumps(meta)
            db.add(user_msg)
            db.commit()
            return "unauthorized"


        # Initialize GitHub Agent if enabled
        github_agent = None
        logger.info(f"github_settings.enabled: {github_settings.enabled}")
        if github_settings.enabled and github_settings.token:
            model_name = os.getenv("OPENAI_MODEL", "qwen-plus")
            llm = ChatOpenAI(model=model_name, api_key=settings.OPENAI_API_KEY, api_base=settings.OPENAI_API_BASE_URL)
            client = GitHubClient(token=github_settings.token, base_url=github_settings.base_url)
            server = GitHubMCPServer(github_client=client)
            github_agent = GitHubAgent(
                llm=llm,
                mcp_server=server,
                github_client=client,
                default_repo=github_settings.default_repo
            )

        agent = FinanceAgent(github_agent=github_agent)
        logger.info("Starting FinanceAgent run")
        try:
            reply = agent.run(user_msg.content, [], thread_id=str(conv.id))
            ai_msg = Message(conversation_id=conv.id, role="assistant", content=reply, meta=json.dumps({"parent_message_id": message_id}))
            db.add(ai_msg)
            db.commit()
            db.refresh(ai_msg)
        except Exception as e:
            db.rollback()
            meta = {}
            try:
                meta = json.loads(user_msg.meta) if user_msg.meta else {}
            except Exception:
                meta = {}
            meta.update({"status": "failed", "error": str(e)[:300]})
            user_msg.meta = json.dumps(meta)
            db.add(user_msg)
            db.commit()
            return "error"

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
