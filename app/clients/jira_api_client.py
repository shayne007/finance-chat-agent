import os
import requests
from typing import Dict, Any, Optional


class JiraAPIClient:
    def __init__(self, domain: Optional[str] = None, email: Optional[str] = None, api_token: Optional[str] = None):
        self.base_url = f"https://{domain}.atlassian.net" if domain else None
        self.auth = (email, api_token) if email and api_token else None
        self.headers = {"Accept": "application/json", "Content-Type": "application/json"}

    @classmethod
    def from_env(cls) -> "JiraAPIClient":
        return cls(
            domain=os.getenv("JIRA_DOMAIN"),
            email=os.getenv("JIRA_EMAIL"),
            api_token=os.getenv("JIRA_API_TOKEN"),
        )

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