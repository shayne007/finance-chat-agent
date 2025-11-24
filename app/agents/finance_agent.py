import os
from typing import List, Dict

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from app.agents.jira_agent import JiraAgent


class FinanceAgent:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_key) if (OpenAI and self.openai_key) else None
        self.jira = JiraAgent()

    async def run(self, message: str, history: List[Dict[str, str]]) -> str:
        intent = self.jira.classify_intent(message)
        if intent == "create":
            return self.jira.create_ticket(message)
        if intent == "assess":
            return self.jira.assess_ticket(message)
        if intent == "analyze":
            data = self.jira.analyze_requirement(message)
            return f"Structured requirement\n{data}"
        if not self.client:
            return "I can help with finance chat. Jira intents supported: create, assess, analyze."
        msgs = []
        for m in history:
            msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        msgs.append({"role": "user", "content": message})
        r = self.client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.7)
        return r.choices[0].message.content