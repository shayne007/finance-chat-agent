from typing import List
from app.skills.types import Skill
from app.skills.registry import load_skill

class SkillManager:
    """Manager for agent skills, handling progressive disclosure."""
    
    def __init__(self, skills: List[Skill]):
        """
        Initialize manager with skill definitions.
        
        Args:
            skills: List of Skill objects to make available
        """
        self.skills = skills
        self.skills_prompt = self._build_level1_prompt()
        # Expose the load_skill tool
        self.tools = [load_skill]

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

    def get_system_prompt_addendum(self) -> str:
        """
        Get the text to append to the system prompt.
        """
        return f"""

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
and output formats that you must follow.
"""
