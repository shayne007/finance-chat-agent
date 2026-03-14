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
from app.agents.skills_agent import SkillsAgent
from app.skills.manager import SkillManager
from app.skills.registry import SKILLS_REGISTRY

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
        chat_agent: Optional[ChatAgent] = None,
        jira_agent: Optional[JiraAgent] = None,
        rag_agent: Optional = None,
        github_agent: Optional[GitHubAgent] = None,
        skills_agent: Optional[SkillsAgent] = None,
    ):
        """Initialize the Finance Agent.

        Args:
            chat_agent: Optional Chat Agent instance.
            jira_agent: Optional Jira Agent instance.
            rag_agent: Optional RAG Agent instance.
            github_agent: Optional GitHub Agent instance.
            skills_agent: Optional Skills Agent instance.
        """
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.7,
            api_key=self.openai_key
        ) if self.openai_key else None

        # Initialize sub-agents
        self.jira = jira_agent if jira_agent else JiraAgent()
        self.github_agent = github_agent
        self.rag_agent = rag_agent
        self.chat_agent = chat_agent if chat_agent else ChatAgent()
        self.app = None

        # Initialize skills agent if not provided
        self.skills_agent = skills_agent
        if not self.skills_agent and self.llm:
            self.skills_agent = SkillsAgent(
                model_name=self.model_name,
                temperature=0.7,
                skills=SKILLS_REGISTRY
            )

        # Initialize skills manager for skill-aware responses
        self.skills_manager = SkillManager(SKILLS_REGISTRY)

        logger.info(f"FinanceAgent initialized with GitHub agent: {github_agent is not None}")
        logger.info(f"FinanceAgent initialized with skills: {self.skills_agent is not None}")

    def _detect_agent_type(self, message: str) -> str:
        """Detect which agent should handle the message.

        Args:
            message: The user's message.

        Returns:
            The agent type ("github", "jira", "rag", "chat").
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

        # Default to chat for backward compatibility
        return "chat"

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
        elif self.skills_agent:
            # Use skills agent for queries that don't match specific categories
            # or when complex skills are needed
            result = await self.skills_agent.invoke({
                "input": message,
                "history": history
            }, thread_id=thread_id)

            # Add skills awareness to the response
            if result["metadata"]["skills_used"]:
                skills_info = f"\n\n*Skills used: {', '.join(result['metadata']['skills_used'])}*"
                return result["content"] + skills_info

            return result["content"]
        elif self.llm:
            # Fallback to chat for queries without specific routing
            return await self.chat_agent.process_query(message)
        else:
            return "I don't have the necessary tools to help with that request."
