import os
import json
import re
from typing import Dict, Any, Optional
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class JiraAPIClient:
    def __init__(self, domain: Optional[str], email: Optional[str], api_token: Optional[str]):
        self.domain = domain
        self.base_url = f"https://{domain}.atlassian.net" if domain else None
        self.auth = (email, api_token) if email and api_token else None
        self.headers = {"Accept": "application/json", "Content-Type": "application/json"}

    def is_configured(self) -> bool:
        return bool(self.base_url and self.auth)

    def create_ticket(self, project_key: str, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/rest/api/3/issue"
        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": ticket_data["title"],
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [{
                        "type": "paragraph",
                        "content": [{"type": "text", "text": ticket_data["description"]}]
                    }]
                },
                "issuetype": {"name": ticket_data["ticket_type"]},
                "priority": {"name": ticket_data["priority"]},
                "labels": ticket_data.get("labels", []),
            }
        }
        response = requests.post(url, json=payload, headers=self.headers, auth=self.auth)
        response.raise_for_status()
        return response.json()

    def get_ticket(self, ticket_key: str) -> Dict[str, Any]:
        url = f"{self.base_url}/rest/api/3/issue/{ticket_key}"
        response = requests.get(url, headers=self.headers, auth=self.auth)
        response.raise_for_status()
        return response.json()


class JiraAgent:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(model=self.model_name, temperature=0) if self.openai_key else None
        self.jira = JiraAPIClient(
            domain=os.getenv("JIRA_DOMAIN"),
            email=os.getenv("JIRA_EMAIL"),
            api_token=os.getenv("JIRA_API_TOKEN"),
        )
        self.project_key = os.getenv("JIRA_PROJECT_KEY", "PROJ")

    def classify_intent(self, user_input: str) -> str:
        text = user_input.lower()
        if re.search(r"\bassess\b.*\bticket\b", text) or re.search(r"\bticket\b.*\bassess\b", text):
            return "assess"
        if re.search(r"\bcreate\b.*\b(ticket|jira)\b", text) or text.strip().startswith("create a ticket"):
            return "create"
        if re.search(r"\banalyze\b", text) or re.search(r"\brequirement\b", text):
            return "analyze"
        return "general"

    def _llm_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not self.llm:
            return None
        r = self.llm.invoke([HumanMessage(prompt)])
        try:
            return json.loads(r.content)
        except Exception:
            return None

    def analyze_requirement(self, user_input: str) -> Dict[str, Any]:
        prompt = (
            "You are a Business Analyst. Analyze the requirement and output JSON with keys: "
            "title, description, acceptance_criteria, ticket_type, priority, estimated_story_points, labels, components. "
            "Only output JSON. Requirement: " + user_input
        )
        data = self._llm_json(prompt)
        if data:
            return data
        return {
            "title": user_input[:80] or "New Ticket",
            "description": user_input,
            "acceptance_criteria": [],
            "ticket_type": "Task",
            "priority": "Medium",
            "estimated_story_points": None,
            "labels": ["auto"],
            "components": []
        }

    def create_ticket(self, user_input: str) -> str:
        structured = self.analyze_requirement(user_input)
        if not self.jira.is_configured():
            return "Jira is not configured"
        try:
            res = self.jira.create_ticket(self.project_key, structured)
            return f"Ticket created {res.get('key')} {res.get('self')}"
        except Exception as e:
            return f"Error creating ticket {str(e)}"

    def assess_ticket(self, user_input: str) -> str:
        m = re.search(r"[A-Z]+-\d+", user_input)
        if not m:
            return "Ticket id not found"
        ticket_id = m.group(0)
        if not self.jira.is_configured():
            return "Jira is not configured"
        try:
            data = self.jira.get_ticket(ticket_id)
        except Exception as e:
            return f"Error retrieving ticket {str(e)}"
        prompt = (
            "Assess this Jira ticket and return JSON with keys: completeness_score, clarity_score, "
            "priority_alignment, recommendations, missing_elements. Only JSON. Ticket: " + json.dumps(data)
        )
        assessment = self._llm_json(prompt)
        if not assessment:
            return "Unable to assess"
        lines = [
            f"Completeness {assessment.get('completeness_score')}",
            f"Clarity {assessment.get('clarity_score')}",
            f"Priority {assessment.get('priority_alignment')}",
            "Recommendations:",
            *[f"- {r}" for r in assessment.get('recommendations', [])],
            "Missing:",
            *[f"- {m}" for m in assessment.get('missing_elements', [])],
        ]
        return "\n".join(lines)