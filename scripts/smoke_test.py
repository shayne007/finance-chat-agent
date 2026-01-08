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
        "/api/v1/agent/invoke",
        json={"user_input": "Analyze this requirement: Build a budget dashboard"},
    )
    assert r.status_code == 200, r.text
    resp = r.json()
    print("Agent Output:", resp.get("output"))


if __name__ == "__main__":
    run()
