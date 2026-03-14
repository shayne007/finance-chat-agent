"""
Skill Middleware
LangChain middleware that implements progressive disclosure for agent skills.
Based on the LangChain middleware pattern for skills.
"""

from typing import Any, Callable, List, Dict
from langchain.agents import AgentExecutor, Agent
from langchain_core.runnables import RunnableConfig

from app.skills.types import Skill
from app.skills.registry import load_skill
from app.utils.prompts import build_skills_prompt, get_skills_instructions_template


class SkillMiddleware:
    """Middleware implementing progressive disclosure for agent skills."""

    # Register the load_skill tool so it's available to the agent
    tools = [load_skill]

    def __init__(self, skills: List[Skill]):
        """
        Initialize middleware with skill definitions.

        Args:
            skills: List of Skill objects to make available
        """
        self.skills = skills
        self.skills_prompt = build_skills_prompt(skills)

    def wrap_agent(self, agent: Agent, tools: List) -> AgentExecutor:
        """Wrap an agent with skill middleware."""
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=20
        )
        return executor

    def enhance_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance input with skills information."""
        skills_addendum = get_skills_instructions_template().format(
            skills_prompt=self.skills_prompt
        )

        if "input" in input_data:
            input_data["input"] = f"{input_data['input']}\n\n{skills_addendum}"

        return input_data
        """
        Inject skill descriptions into system prompt before each model call.
        This is the core of progressive disclosure - we show metadata but
        not full content upfront.
        """
        # Build the skills addendum (Level 1: Metadata only)
        skills_addendum = get_skills_instructions_template().format(
            skills_prompt=self.skills_prompt
        )

        # Append to existing system message content
        if hasattr(request, 'messages') and request.messages:
            # Find the system message and append to it
            for message in request.messages:
                if isinstance(message, SystemMessage):
                    if isinstance(message.content, str):
                        # Simple string content
                        new_content = message.content + skills_addendum
                        message.content = new_content
                    elif isinstance(message.content, list):
                        # List content blocks
                        text_blocks = [block for block in message.content if block.get("type") == "text"]
                        if text_blocks:
                            # Append to first text block
                            text_blocks[0]["text"] += skills_addendum
                        else:
                            # Add new text block
                            message.content.append({"type": "text", "text": skills_addendum})

        # Call the model with modified request
        return handler(request, config)