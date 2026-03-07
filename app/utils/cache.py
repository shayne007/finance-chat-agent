"""
Caching utilities for the application
"""

from functools import lru_cache
from typing import List
from app.skills.types import Skill


@lru_cache(maxsize=10)
def get_cached_skills_prompt(skills_tuple: tuple) -> str:
    """Get cached skills prompt for a tuple of skills.

    Args:
        skills_tuple: Tuple of Skill objects (hashable)

    Returns:
        Skills prompt string
    """
    skills_list = list(skills_tuple)
    from app.utils.prompts import build_skills_prompt
    return build_skills_prompt(skills_list)


@lru_cache(maxsize=5)
def get_cached_skill_manager(skills_tuple: tuple):
    """Get cached SkillManager instance.

    Args:
        skills_tuple: Tuple of Skill objects (hashable)

    Returns:
        SkillManager instance
    """
    skills_list = list(skills_tuple)
    from app.skills.manager import SkillManager
    return SkillManager(skills_list)


def skills_to_tuple(skills: List) -> tuple:
    """Convert skills list to tuple for caching."""
    return tuple((skill.name, skill.description, skill.category, skill.content) for skill in skills)