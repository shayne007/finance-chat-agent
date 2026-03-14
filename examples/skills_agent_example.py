"""Example usage of the Skills-based Agent with SkillMiddleware.

This script demonstrates how to use the SkillsAgent with progressive disclosure
to analyze code and generate documentation.
"""

import asyncio
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_basic_skills_usage():
    """Basic example of using skills."""
    print("\n=== Basic Skills Usage Example ===")

    from app.config.agents import AgentConfig

    # Create skills agent
    skills_agent = AgentConfig.create_skills_only_agent()

    # Simple code analysis task
    result = await skills_agent.invoke({
        "input": "Analyze this Python code and identify classes, functions, and dependencies:\n\nclass Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b"
    })

    print(f"Response: {result['content']}")
    print(f"Skills used: {result['metadata']['skills_used']}")


async def example_finance_agent_with_skills():
    """Example of using FinanceAgent with skills support."""
    print("\n=== Finance Agent with Skills Example ===")

    from app.config.agents import AgentConfig

    # Create finance agent with skills
    finance_agent = AgentConfig.create_finance_agent(enable_skills=True)

    # Example query that would trigger skills loading
    result = await finance_agent.run(
        "Analyze the code structure of this repository and generate documentation with architecture diagrams. "
        "The repository contains Python and Java files with microservices architecture.",
        thread_id="example-thread-123"
    )

    print(f"Response: {result}")
    print("Note: This example shows how skills are integrated into the main agent flow.")


async def example_skill_progressive_disclosure():
    """Demonstrate progressive disclosure with skills."""
    print("\n=== Progressive Disclosure Example ===")

    from app.agents.skills_agent import SkillsAgent

    # Create agent
    skills_agent = SkillsAgent()

    # First query - should trigger skill loading
    print("Query 1: Generate documentation for a Python repository")
    result1 = await skills_agent.invoke({
        "input": "Generate documentation for a Python repository with REST API endpoints"
    })

    print(f"Skills used in first query: {result1['metadata']['skills_used']}")

    # Second query - might reuse loaded skills
    print("\nQuery 2: Add API endpoint documentation to the existing docs")
    result2 = await skills_agent.invoke({
        "input": "Now add detailed API endpoint documentation to the generated docs"
    })

    print(f"Skills used in second query: {result2['metadata']['skills_used']}")


async def example_skill_monitoring():
    """Example of tracking skill usage and performance."""
    print("\n=== Skill Monitoring Example ===")

    from app.agents.skills_agent import SkillsAgent

    # Create agent
    skills_agent = SkillsAgent()

    # Track skill usage across multiple queries
    queries = [
        "Analyze Python code structure",
        "Generate documentation for the codebase",
        "Create architecture diagrams",
        "Extract database schema from SQL files"
    ]

    skills_used = []

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        result = await skills_agent.invoke({"input": query})
        skills_used.extend(result['metadata']['skills_used'])
        print(f"  Skills loaded: {result['metadata']['skills_used']}")

    # Print summary
    from collections import Counter
    skill_counts = Counter(skills_used)
    print(f"\nSkill Usage Summary:")
    for skill, count in skill_counts.items():
        print(f"  {skill}: {count} times")


async def main():
    """Run all examples."""
    print("Skills Agent Examples")
    print("===================")

    try:
        await example_basic_skills_usage()
        await example_finance_agent_with_skills()
        await example_skill_progressive_disclosure()
        await example_skill_monitoring()

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"\nError: {e}")
        print("Note: Make sure you have set your OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    asyncio.run(main())