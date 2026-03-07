"""
Skill Middleware
LangChain middleware that implements progressive disclosure for agent skills.
Based on the LangChain middleware pattern for skills.
"""

from typing import Callable, List
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    AgentMiddleware
)
from langchain.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from app.skills.types import Skill
from app.skills.registry import load_skill


class SkillMiddleware(AgentMiddleware):
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
        self.skills_prompt = self._build_level1_prompt()

    def _build_level1_prompt(self) -> str:
        """
        Build Level 1 metadata prompt (lightweight skill descriptions).
        This is injected into the system prompt.
        """
        skills_by_category = {}
        for skill in self.skills:
            if skill.category not in skills_by_category:
                skills_by_category[skill.category] = []
            skills_by_category[skill.category].append(skill)

        prompt_parts = []
        for category, category_skills in sorted(skills_by_category.items()):
            prompt_parts.append(f"\n**{category.title()} Skills:**")
            for skill in category_skills:
                prompt_parts.append(f"- **{skill.name}**: {skill.description}")

        return "\n".join(prompt_parts)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
        config: RunnableConfig,
    ) -> ModelResponse:
        """
        Inject skill descriptions into system prompt before each model call.
        This is the core of progressive disclosure - we show metadata but
        not full content upfront.
        """
        # Build the skills addendum (Level 1: Metadata only)
        skills_addendum = f"""

## Available Skills

You have access to specialized skills that provide detailed instructions for specific tasks.
Each skill contains comprehensive guidelines, tools, examples, and best practices.

{self.skills_prompt}

**How to Use Skills:**

1. **Identify Relevant Skills**: Determine which skill(s) are needed for the user's request
2. **Load Full Instructions**: Call `load_skill(skill_name)` to get detailed content
3. **Follow Guidelines**: Use the loaded skill's instructions to complete the task
4. **Load Multiple Skills**: You can load several skills for complex multi-step tasks

**Important:** Don't try to guess skill contents. Always load the skill to get accurate,
detailed instructions. The skill content includes specific tools, business logic rules,
examples, and best practices that ensure high-quality results.

**Pro Tip:** Skills are cached after loading, so you can reference them throughout the conversation.
"""

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