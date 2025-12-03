import os
from typing import List, Dict, TypedDict, Annotated, Sequence, Optional
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.redis import RedisSaver
from redis import Redis
from app.core.config import settings
from app.agents.jira_agent import JiraAgent


class FinanceAgent:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(model=self.model_name, temperature=0.7) if self.openai_key else None
        self.jira = JiraAgent()
        self.app = None
        if self.llm:
            try:
                redis_url = settings.REDIS_CHECKPOINT_URL or os.getenv("REDIS_CHECKPOINT_URL") or "redis://localhost:6379"
                client = Redis.from_url(redis_url, decode_responses=True)

                class AgentState(TypedDict):
                    messages: Annotated[Sequence[BaseMessage], operator.add]

                def llm_node(state: AgentState):
                    r = self.llm.invoke(state["messages"])  # sync inside graph
                    return {"messages": [r]}

                graph = StateGraph(AgentState)
                graph.add_node("llm", llm_node)
                graph.set_entry_point("llm")
                checkpointer = RedisSaver(client)
                self.app = graph.compile(checkpointer=checkpointer)
            except Exception:
                self.app = None

    def run(self, message: str, history: List[Dict[str, str]], thread_id: Optional[str] = None) -> str:
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
        if self.app:
            try:
                state = self.app.invoke(
                    {"messages": [HumanMessage(message)]},
                    config={"configurable": {"thread_id": thread_id or "default"}},
                )
                resp = state.get("messages", [])
                return resp[-1].content if resp else ""
            except Exception:
                pass
        msgs = []
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "assistant":
                msgs.append(AIMessage(content))
            else:
                msgs.append(HumanMessage(content))
        msgs.append(HumanMessage(message))
        r = self.llm.invoke(msgs)
        return r.content
