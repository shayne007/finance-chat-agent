import os, sys
sys.path.append(os.path.abspath("."))
from fastapi.testclient import TestClient
from uuid import UUID
from time import sleep
from app.main import app

# Ensure no external LLM calls during smoke test
os.environ.pop("OPENAI_API_KEY", None)


def run():
    client = TestClient(app)
    user_id = "00000000-0000-0000-0000-000000000001"

    r = client.post("/api/v1/conversations/", params={"user_id": user_id}, json={"title": "My Finance Chat"})
    assert r.status_code == 201, r.text
    conv = r.json()
    conv_id = conv["id"]
    UUID(conv_id)

    r = client.post(
        "/api/v1/messages/chat-request",
        params={"user_id": user_id},
        json={"message": "Analyze this requirement: Build a budget dashboard", "stream": False, "conversation_id": conv_id},
    )
    assert r.status_code == 200, r.text
    queued = r.json()
    mid = queued["message_id"]
    for _ in range(20):
        s = client.get(f"/api/v1/messages/chat-request/{mid}", params={"user_id": user_id})
        assert s.status_code == 200, s.text
        data = s.json()
        if data.get("status") == "completed":
            print("Agent Output:", data.get("content"))
            break
        sleep(0.2)


if __name__ == "__main__":
    run()
