"""
Shared utilities for agent implementations
"""

import uuid
from typing import List, Dict, Any, Optional
from app.skills.types import Skill
from app.skills.manager import SkillManager
from langchain_openai import ChatOpenAI


def create_llm(
    model_name: str = "gpt-4o",
    temperature: float = 0.7,
    **kwargs
) -> ChatOpenAI:
    """Create a ChatOpenAI instance with common configuration."""
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        **kwargs
    )


def extract_skills_from_result(result: str) -> List[str]:
    """Extract skill names from a result string using pattern matching."""
    import re
    pattern = r"Loaded Skill: (\w+)"
    matches = re.findall(pattern, result)
    return matches


def generate_thread_id(prefix: str = "skills") -> str:
    """Generate a unique thread ID with a prefix."""
    return f"{prefix}-{uuid.uuid4().hex}"


def create_agent_response(
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a standardized agent response format."""
    response = {
        "content": content,
        "type": "agent",
        "metadata": metadata or {}
    }
    return response


def create_skills_manager(skills: List[Skill]) -> SkillManager:
    """Create a SkillManager instance with caching."""
    return SkillManager(skills)