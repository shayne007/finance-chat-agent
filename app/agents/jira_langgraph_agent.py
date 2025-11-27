import os
import json
from typing import TypedDict, Literal, Optional, Dict, Any

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.clients.jira_api_client import JiraAPIClient


class AgentState(TypedDict):
    user_input: str
    intent: Optional[Literal["analyze", "create", "assess"]]
    structured_output: Optional[dict]
    jira_ticket_id: Optional[str]
    api_response: Optional[dict]
    final_output: str
    error: Optional[str]
    agent_id: Optional[str]
    execution_id: Optional[str]


INTENT_CLASSIFICATION_PROMPT = (
    "Analyze the user's request and classify the intent.\n\n"
    "User Request: {user_input}\n\n"
    "Respond ONLY with a JSON object:\n"
    "{\n  \"intent\": \"analyze\" | \"create\" | \"assess\",\n  \"reasoning\": \"brief explanation\"\n}"
)

REQUIREMENT_ANALYSIS_PROMPT = (
    "You are a Business Analyst. Analyze the requirement and output a structured ticket specification.\n\n"
    "User Requirement: {user_input}\n\n"
    "Respond ONLY with valid JSON matching this schema:\n"
    "{\n  \"title\": \"string\",\n  \"description\": \"string\",\n  \"acceptance_criteria\": [\"string\"],\n  \"ticket_type\": \"Story\" | \"Bug\" | \"Task\" | \"Epic\",\n  \"priority\": \"Highest\" | \"High\" | \"Medium\" | \"Low\" | \"Lowest\",\n  \"estimated_story_points\": integer (1-13) or null,\n  \"labels\": [\"string\"],\n  \"components\": [\"string\"]\n}"
)

TICKET_ASSESSMENT_PROMPT = (
    "Assess this Jira ticket for quality and completeness.\n\n"
    "Ticket Data: {ticket_data}\n\n"
    "Respond ONLY with valid JSON:\n"
    "{\n  \"ticket_id\": \"string\",\n  \"completeness_score\": float (0-10),\n  \"clarity_score\": float (0-10),\n  \"priority_alignment\": \"string\",\n  \"recommendations\": [\"string\"],\n  \"missing_elements\": [\"string\"]\n}"
)


def _extract_json(text: str) -> dict:
    t = text.strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0]
    return json.loads(t.strip())


class JiraLangGraphAgent:
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=self.model, temperature=0.3) if api_key else None
        self.jira = JiraAPIClient.from_env()
        self.project_key = os.getenv("JIRA_PROJECT_KEY", "PROJ")
        self.workflow = None
        self.app = None
        if LANGGRAPH_AVAILABLE:
            self._build_workflow()

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        if not self.llm:
            raise RuntimeError("OpenAI client not configured")
        msg = [HumanMessage(prompt)]
        r = self.llm.invoke(msg)
        content = r.content
        return _extract_json(content)

    def _classify(self, state: AgentState) -> AgentState:
        if not self.llm:
            text = state["user_input"].lower()
            if "assess" in text and "ticket" in text:
                state["intent"] = "assess"
            elif text.strip().startswith("create a ticket") or ("create" in text and ("ticket" in text or "jira" in text)):
                state["intent"] = "create"
            elif "analyze" in text or "requirement" in text:
                state["intent"] = "analyze"
            else:
                state["intent"] = None
            return state
        result = self._call_llm(INTENT_CLASSIFICATION_PROMPT.format(user_input=state["user_input"]))
        state["intent"] = result.get("intent")
        return state

    def _analyze(self, state: AgentState) -> AgentState:
        if not self.llm:
            data = {
                "title": state["user_input"][:80] or "New Ticket",
                "description": state["user_input"],
                "acceptance_criteria": [],
                "ticket_type": "Task",
                "priority": "Medium",
                "estimated_story_points": None,
                "labels": ["auto"],
                "components": [],
            }
        else:
            data = self._call_llm(REQUIREMENT_ANALYSIS_PROMPT.format(user_input=state["user_input"]), temperature=0.5)
        state["structured_output"] = data
        state["final_output"] = json.dumps(data, indent=2)
        return state

    def _create(self, state: AgentState) -> AgentState:
        data = self._call_llm(REQUIREMENT_ANALYSIS_PROMPT.format(user_input=state["user_input"]), temperature=0.5)
        try:
            api_res = self.jira.create_ticket(self.project_key, data)
            state["jira_ticket_id"] = api_res.get("key")
            state["api_response"] = api_res
            state["final_output"] = f"Ticket created: {api_res.get('key')}\n{api_res.get('self')}"
        except Exception as e:
            state["error"] = str(e)
            state["final_output"] = f"Error creating ticket: {str(e)}"
        return state

    def _assess(self, state: AgentState) -> AgentState:
        parts = state["user_input"].split()
        ticket_id = parts[-1] if parts else ""
        try:
            ticket_data = self.jira.get_ticket(ticket_id)
            if not self.llm:
                assessment = {
                    "ticket_id": ticket_id,
                    "completeness_score": 5.0,
                    "clarity_score": 5.0,
                    "priority_alignment": "Unknown (no AI configured)",
                    "recommendations": ["Configure OPENAI_API_KEY for full assessment"],
                    "missing_elements": [],
                }
            else:
                assessment = self._call_llm(
                    TICKET_ASSESSMENT_PROMPT.format(ticket_data=json.dumps(ticket_data, indent=2)),
                    temperature=0.5,
                )
            state["structured_output"] = assessment
            lines = [
                f"Completeness: {assessment.get('completeness_score')}/10",
                f"Clarity: {assessment.get('clarity_score')}/10",
                f"Priority Alignment: {assessment.get('priority_alignment')}",
                "Recommendations:",
                *[f"• {r}" for r in assessment.get("recommendations", [])],
                "Missing Elements:",
                *[f"• {m}" for m in assessment.get("missing_elements", [])],
            ]
            state["final_output"] = "\n".join(lines)
        except Exception as e:
            state["error"] = str(e)
            state["final_output"] = f"Error assessing ticket: {str(e)}"
        return state

    def _route(self, state: AgentState) -> str:
        m = {
            "analyze": "analyze_node",
            "create": "create_node",
            "assess": "assess_node",
        }
        return m.get(state.get("intent") or "", END)

    def _build_workflow(self) -> None:
        workflow = StateGraph(AgentState)
        workflow.add_node("classify", self._classify)
        workflow.add_node("analyze_node", self._analyze)
        workflow.add_node("create_node", self._create)
        workflow.add_node("assess_node", self._assess)
        workflow.set_entry_point("classify")
        workflow.add_conditional_edges("classify", self._route)
        workflow.add_edge("analyze_node", END)
        workflow.add_edge("create_node", END)
        workflow.add_edge("assess_node", END)
        self.workflow = workflow
        self.app = workflow.compile()

    def invoke(self, user_input: str) -> AgentState:
        state: AgentState = {
            "user_input": user_input,
            "intent": None,
            "structured_output": None,
            "jira_ticket_id": None,
            "api_response": None,
            "final_output": "",
            "error": None,
            "agent_id": None,
            "execution_id": None,
        }
        if LANGGRAPH_AVAILABLE and self.app:
            return self.app.invoke(state)
        # Fallback sequential execution when LangGraph unavailable
        state = self._classify(state)
        intent = state.get("intent")
        if intent == "analyze":
            return self._analyze(state)
        if intent == "create":
            return self._create(state)
        if intent == "assess":
            return self._assess(state)
        state["final_output"] = "No actionable intent detected"
        return state