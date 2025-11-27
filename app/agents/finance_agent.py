import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.jira_agent import JiraAgent


class FinanceAgent:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(model=self.model_name, temperature=0.7) if self.openai_key else None
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
        if not self.llm:
            return "I can help with finance chat. Jira intents supported: create, assess, analyze."
        msgs = []
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "assistant":
                msgs.append(AIMessage(content))
            else:
                msgs.append(HumanMessage(content))
        msgs.append(HumanMessage(message))
        r = await self.llm.ainvoke(msgs)
        return r.content