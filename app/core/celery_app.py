from celery import Celery
import os
from app.core.config import settings

celery_app = Celery(
    "finance_chat_agent",
    broker=os.getenv("CELERY_BROKER_URL", settings.CELERY_BROKER_URL),
    backend=os.getenv("CELERY_RESULT_BACKEND", settings.CELERY_RESULT_BACKEND),
)

celery_app.conf.task_always_eager = settings.CELERY_TASK_ALWAYS_EAGER
celery_app.conf.imports = [
    "app.tasks.message_tasks",
]