from dataclasses import dataclass
from typing import List, Callable, Optional, Any

@dataclass
class Skill:
    """A skill that can be progressively disclosed to the agent."""
    name: str                     # Unique identifier  
    description: str              # Brief description (Level 1)
    content: str                  # Full instructions (Level 2)
    tools: List[Callable] = None  # Optional skill-specific tools
    category: str = "general"     # Skill category
    token_budget: int = 2000      # Estimated tokens
