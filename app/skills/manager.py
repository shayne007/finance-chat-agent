from typing import List
from app.skills.types import Skill
from app.skills.registry import load_skill
from app.utils.prompts import build_skills_prompt, get_skills_instructions_template

class SkillManager:
    """Manager for agent skills, handling progressive disclosure."""
    
    def __init__(self, skills: List[Skill]):
        """
        Initialize manager with skill definitions.
        
        Args:
            skills: List of Skill objects to make available
        """
        self.skills = skills
        self.skills_prompt = build_skills_prompt(skills)
        # Expose the load_skill tool
        self.tools = [load_skill]

    
    def get_system_prompt_addendum(self) -> str:
        """
        Get the text to append to the system prompt.
        """
        return get_skills_instructions_template().format(
            skills_prompt=self.skills_prompt
        )
