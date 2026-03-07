"""Agent configuration management.

This module provides configuration and initialization for all agent types.
"""

from typing import Dict, Any, Optional
import logging

from app.agents.finance_agent import FinanceAgent
from app.agents.jira_agent import JiraAgent
from app.agents.github_agent import GitHubAgent
from app.agents.rag_agent import RAGAgent
from app.agents.skills_agent import SkillsAgent
from app.skills.manager import SkillManager
from app.skills.registry import SKILLS_REGISTRY
from app.core.config import settings

logger = logging.getLogger(__name__)


class AgentConfig:
    """Configuration and factory for all agent types."""

    @staticmethod
    def create_finance_agent(
        jira_agent: Optional[JiraAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
        github_agent: Optional[GitHubAgent] = None,
        skills_agent: Optional[SkillsAgent] = None,
        enable_skills: bool = True,
    ) -> FinanceAgent:
        """Create a FinanceAgent with all sub-agents.

        Args:
            jira_agent: Optional JiraAgent instance
            rag_agent: Optional RAGAgent instance
            github_agent: Optional GitHubAgent instance
            skills_agent: Optional SkillsAgent instance
            enable_skills: Whether to enable skills support

        Returns:
            Configured FinanceAgent instance
        """
        logger.info("Creating FinanceAgent with skills support enabled")

        # Create skills agent if enabled and not provided
        if enable_skills and not skills_agent:
            skills_agent = SkillsAgent(
                model_name=settings.OPENAI_MODEL,
                temperature=0.7,
                skills=SKILLS_REGISTRY
            )

        # Create main finance agent
        finance_agent = FinanceAgent(
            jira_agent=jira_agent,
            rag_agent=rag_agent,
            github_agent=github_agent,
            skills_agent=skills_agent
        )

        return finance_agent

    @staticmethod
    def create_skills_only_agent(
        model_name: str = None,
        temperature: float = 0.7,
        skills: Optional[list] = None,
    ) -> SkillsAgent:
        """Create a Skills-only agent.

        Args:
            model_name: Model name to use
            temperature: Temperature setting
            skills: Optional list of skills to use

        Returns:
            Configured SkillsAgent instance
        """
        logger.info("Creating Skills-only agent")

        return SkillsAgent(
            model_name=model_name or settings.OPENAI_MODEL,
            temperature=temperature,
            skills=skills or SKILLS_REGISTRY
        )

    @staticmethod
    def get_agent_info(agent: Any) -> Dict[str, Any]:
        """Get information about an agent's capabilities.

        Args:
            agent: Agent instance

        Returns:
            Dictionary with agent information
        """
        if isinstance(agent, FinanceAgent):
            return {
                "type": "FinanceAgent",
                "sub_agents": {
                    "jira": agent.jira is not None,
                    "github": agent.github_agent is not None,
                    "rag": agent.rag_agent is not None,
                    "skills": agent.skills_agent is not None,
                },
                "skills_count": len(SKILLS_REGISTRY) if agent.skills_manager else 0
            }
        elif isinstance(agent, SkillsAgent):
            return {
                "type": "SkillsAgent",
                "skills_count": len(agent.skills),
                "model": agent.model_name,
                "temperature": agent.temperature
            }
        elif isinstance(agent, GitHubAgent):
            return {
                "type": "GitHubAgent",
                "has_mcp": True
            }
        elif isinstance(agent, JiraAgent):
            return {
                "type": "JiraAgent",
                "intent_classification": True
            }
        else:
            return {
                "type": type(agent).__name__
            }


# Factory function for easy agent creation
def create_agent(
    agent_type: str = "finance",
    **kwargs
) -> Any:
    """Factory function to create agents.

    Args:
        agent_type: Type of agent to create ("finance", "skills", "github", "jira")
        **kwargs: Additional arguments for agent creation

    Returns:
        Configured agent instance
    """
    if agent_type == "finance":
        return AgentConfig.create_finance_agent(**kwargs)
    elif agent_type == "skills":
        return AgentConfig.create_skills_only_agent(**kwargs)
    elif agent_type == "github":
        return GitHubAgent(**kwargs)
    elif agent_type == "jira":
        return JiraAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# Example usage and initialization
def initialize_default_agents() -> Dict[str, Any]:
    """Initialize all default agents.

    Returns:
        Dictionary with all initialized agents
    """
    logger.info("Initializing default agents")

    # Initialize skills manager
    skills_manager = SkillManager(SKILLS_REGISTRY)

    # Create agents
    agents = {
        "finance": AgentConfig.create_finance_agent(enable_skills=True),
        "skills": AgentConfig.create_skills_only_agent(),
    }

    # Add agent information
    agent_info = {
        name: AgentConfig.get_agent_info(agent)
        for name, agent in agents.items()
    }

    logger.info(f"Initialized agents: {list(agents.keys())}")
    logger.info(f"Skills available: {len(SKILLS_REGISTRY)}")

    return {
        "agents": agents,
        "info": agent_info,
        "skills_manager": skills_manager
    }


if __name__ == "__main__":
    # Example usage
    agents = initialize_default_agents()

    # Print agent information
    for name, info in agents["info"].items():
        print(f"{name}: {info}")