import os, sys
sys.path.append(os.path.abspath("."))
from fastapi.testclient import TestClient
from uuid import UUID
from app.main import app


def run():
    client = TestClient(app)
    user_id = "00000000-0000-0000-0000-000000000001"

    r = client.post("/api/v1/conversations/", params={"user_id": user_id}, json={"title": "My Finance Chat"})
    assert r.status_code == 201, r.text
    conv = r.json()
    conv_id = conv["id"]
    UUID(conv_id)

    r = client.post(
        "/api/v1/messages/",
        params={"user_id": user_id},
        json={"message": "How should I budget?", "conversation_id": conv_id},
    )
    assert r.status_code == 200, r.text
    resp = r.json()
    UUID(resp["message_id"])
    print("Conversation:", conv_id)
    print("Suggestion:", resp["content"])


if __name__ == "__main__":
    run()