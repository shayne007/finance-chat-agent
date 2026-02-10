from langchain.tools import tool
from typing import List
from app.skills.types import Skill
from app.skills.definitions.code_analysis import CODE_ANALYSIS_SKILL

# Global skills registry
SKILLS_REGISTRY: List[Skill] = [
    CODE_ANALYSIS_SKILL,
    # Add other skills here
]

@tool
def load_skill(skill_name: str) -> str:
    """Load the full content and instructions for a specific skill.
    
    Use this when you need detailed information about how to perform a specific
    type of task. This provides comprehensive instructions, guidelines, tools,
    examples, and best practices for that skill area.
    
    Available skills:
    - code_analysis: Analyze code structure and extract entities
    
    Args:
        skill_name: The name of the skill to load (e.g., "code_analysis")
        
    Returns:
        Full skill content with detailed instructions, tools, and examples
    """
    # Find the requested skill
    for skill in SKILLS_REGISTRY:
        if skill.name == skill_name:
            return f"""# Loaded Skill: {skill.name}

Category: {skill.category}
Estimated Token Budget: {skill.token_budget}

{skill.content}

---

**You now have access to this skill's full capabilities. Use the instructions above to complete your task.**
"""
    
    # Skill not found
    available = ", ".join(s.name for s in SKILLS_REGISTRY)
    return f"Skill '{skill_name}' not found. Available skills: {available}"
