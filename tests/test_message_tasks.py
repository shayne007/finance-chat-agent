import logging


def test_process_message_task_logs_and_returns_ai_message_id(monkeypatch, caplog):
    import app.tasks.message_tasks as message_tasks

    class DummyColumn:
        def __init__(self, name: str):
            self.name = name

        def __eq__(self, other):
            return (self.name, "==", other)

    class DummyMessage:
        id = DummyColumn("id")
        conversation_id = DummyColumn("conversation_id")

        def __init__(self, conversation_id=None, role=None, content=None, meta=None):
            self.id = None
            self.conversation_id = conversation_id
            self.role = role
            self.content = content
            self.meta = meta

    class DummyConversation:
        id = DummyColumn("id")
        user_id = DummyColumn("user_id")

        def __init__(self, id=None, user_id=None):
            self.id = id
            self.user_id = user_id

    user_msg = DummyMessage(conversation_id="conv-1", role="user", content="hi", meta=None)
    conv = DummyConversation(id="conv-1", user_id="user-1")

    class DummyQuery:
        def __init__(self, result):
            self._result = result

        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return self._result

    class DummyDB:
        def query(self, model):
            if model is message_tasks.Message:
                return DummyQuery(user_msg)
            if model is message_tasks.Conversation:
                return DummyQuery(conv)
            raise AssertionError(f"Unexpected model: {model}")

        def add(self, obj):
            return None

        def commit(self):
            return None

        def refresh(self, obj):
            if getattr(obj, "id", None) is None:
                obj.id = "ai-123"

        def rollback(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(message_tasks, "SessionLocal", lambda: DummyDB())
    monkeypatch.setattr(message_tasks, "Message", DummyMessage)
    monkeypatch.setattr(message_tasks, "Conversation", DummyConversation)

    class DummyAgent:
        def __init__(self, github_agent=None):
            self.github_agent = github_agent

        def run(self, content, history, thread_id):
            return "reply"

    monkeypatch.setattr(message_tasks, "FinanceAgent", DummyAgent)

    monkeypatch.setattr(message_tasks.github_settings, "enabled", False, raising=False)
    monkeypatch.setattr(message_tasks.github_settings, "token", "", raising=False)

    with caplog.at_level(logging.INFO):
        result = message_tasks.process_message_task("msg-1", "user-1")

    assert result == "ai-123"
    assert any("github_settings.enabled:" in record.message for record in caplog.records)
