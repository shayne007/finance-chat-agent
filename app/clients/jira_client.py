import requests
from jira import JIRA

class JiraClient:
    def __init__(self, server_url, username, api_token):
        self.client = JIRA(
            server=server_url,
            basic_auth=(username, api_token)
        )
    
    def create_issue(self, fields: dict):
        """Create Jira issue with validation"""
        required_fields = ['project', 'summary', 'issuetype']
        for field in required_fields:
            if field not in fields:
                raise ValueError(f"Missing required field: {field}")
        
        return self.client.create_issue(fields=fields)
    
    def get_issue(self, issue_key: str):
        """Retrieve issue with error handling"""
        try:
            return self.client.issue(issue_key)
        except Exception as e:
            raise ValueError(f"Ticket {issue_key} not found: {str(e)}")
    
    def search_issues(self, jql: str, max_results: int = 50):
        """Search issues using JQL"""
        return self.client.search_issues(jql, maxResults=max_results)