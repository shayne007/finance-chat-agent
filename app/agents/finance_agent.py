import os
from typing import List, Dict, TypedDict, Annotated, Sequence, Optional
import operator

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.redis import RedisSaver
from redis import Redis

from app.core.config import settings
from app.agents.jira_agent import JiraAgent
from app.agents.github_agent import GitHubAgent

logger = logging.getLogger(__name__)


class FinanceAgent:
    """Finance Agent with GitHub routing integration.

    This agent routes queries to the appropriate sub-agent (Jira, RAG, GitHub)
    based on keyword detection in the user's query.
    """

    # GitHub keywords for routing detection
    GITHUB_KEYWORDS: List[str] = [
        "github", "issue", "pr", "pull request", "repo",
        "commit", "branch", "file", "code", "search",
    ]

    def __init__(
        self,
        jira_agent: Optional[JiraAgent] = None,
        rag_agent: Optional = None,
        github_agent: Optional[GitHubAgent] = None,
    ):
        """Initialize the Finance Agent.

        Args:
            jira_agent: Optional Jira Agent instance.
            rag_agent: Optional RAG Agent instance.
            github_agent: Optional GitHub Agent instance.
        """
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(model=self.model_name, temperature=0.7) if self.openai_key else None
        self.jira = jira_agent if jira_agent else JiraAgent()
        self.github_agent = github_agent
        self.rag_agent = rag_agent
        self.app = None

        logger.info(f"FinanceAgent initialized with GitHub agent: {github_agent is not None}")

    def _detect_agent_type(self, message: str) -> str:
        """Detect which agent should handle the message.

        Args:
            message: The user's message.

        Returns:
            The agent type ("github", "jira", "rag", "unknown").
        """
        message_lower = message.lower()

        # Check for GitHub keywords
        if any(keyword in message_lower for keyword in self.GITHUB_KEYWORDS):
            return "github"

        # Check for Jira keywords (existing functionality)
        jira_keywords = ["jira", "ticket", "bug", "sprint"]
        if any(keyword in message_lower for keyword in jira_keywords):
            return "jira"

        # Check for RAG keywords (existing functionality)
        rag_keywords = ["search", "document", "report", "analyze"]
        if any(keyword in message_lower for keyword in rag_keywords):
            return "rag"

        # Default to Jira for backward compatibility
        return "jira"

    async def run(
        self, message: str, history: List[Dict[str, str]] = [], thread_id: Optional[str] = None
    ) -> str:
        """Process a user message and return a response.

        Args:
            message: The user's message.
            history: Conversation history.
            thread_id: Thread ID for state management.

        Returns:
            The agent's response.
        """
        # Detect which agent should handle this query
        agent_type = self._detect_agent_type(message)

        logger.info(f"Routing query to {agent_type}_agent")

        # Route to the appropriate agent
        if agent_type == "github" and self.github_agent:
            return await self.github_agent.process_query(message)
        elif agent_type == "jira" and self.jira:
            intent = self.jira.classify_intent(message)
            if intent == "create":
                return self.jira.create_ticket(message)
            elif intent == "assess":
                data = self.jira.analyze_requirement(message)
                return f"Structured requirement\n{data}"
            elif intent == "analyze":
                return self.jira.analyze_requirement(message)
            else:
                return self.jira.create_ticket(message)
        elif agent_type == "rag" and self.rag_agent:
            return f"Searching for information about: {message}"
        elif self.llm:
            # Fallback to Jira for queries without specific routing
            intent = self.jira.classify_intent(message)
            if intent == "create":
                return self.jira.create_ticket(message)
            elif intent == "assess":
                data = self.jira.analyze_requirement(message)
                return f"Structured requirement\n{data}"
            elif intent == "analyze":
                return self.jira.analyze_requirement(message)
            else:
                return self.jira.create_ticket(message)
        else:
            return "I don't have the necessary tools to help with that request."
