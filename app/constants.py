"""
Application constants
"""

# Skill-related constants
class SkillMessages:
    """Constants for skill-related messages."""
    LOADED_SKILL_PATTERN = r"Loaded Skill: (\w+)"
    AVAILABLE_SKILLS = "Available Skills"
    LEVEL_1 = "Level 1"
    LEVEL_2 = "Level 2"

    HOW_TO_USE_SKILLS = """
**How to Use Skills:**

1. **Identify Relevant Skills**: Determine which skill(s) are needed for the user's request
2. **Load Full Instructions**: Call `load_skill(skill_name)` to get detailed content
3. **Follow Guidelines**: Use the loaded skill's instructions to complete the task
4. **Load Multiple Skills**: You can load several skills for complex multi-step tasks

**Important:** Don't try to guess skill contents. Always load the skill to get accurate,
detailed instructions. The skill content includes specific tools, business logic rules,
and output formats that you must follow.

**Pro Tip:** Skills are cached after loading, so you can reference them throughout the conversation.
"""

class AgentConfigDefaults:
    """Default values for agent configuration."""
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.7
    MAX_ITERATIONS = 20
    TOKEN_BUDGET = 2000


class ErrorMessages:
    """Standardized error messages."""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    NOT_FOUND = "not_found"

    @classmethod
    def format(cls, operation: str, error: str) -> str:
        """Format a standardized error message."""
        return f"{operation} failed: {error}"