"""Skills-based Agent that uses SkillMiddleware for progressive disclosure.

This agent uses LangChain with SkillMiddleware to enable on-demand skill loading,
reducing token consumption while maintaining full capabilities.
"""

from typing import List, Dict, Optional, Any
import logging
import uuid

from langchain.agents import create_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage

from app.skills.middleware import SkillMiddleware
from app.skills.registry import SKILLS_REGISTRY, load_skill
from app.skills.manager import SkillManager
from app.core.config import settings
from app.utils.agents import create_llm, create_agent_response
from app.constants import AgentConfigDefaults, SkillMessages
from app.utils.errors import handle_agent_error

logger = logging.getLogger(__name__)


class SkillsAgent:
    """Agent that uses SkillMiddleware for progressive skill disclosure."""

    def __init__(
        self,
        model_name: str = AgentConfigDefaults.DEFAULT_MODEL,
        temperature: float = AgentConfigDefaults.DEFAULT_TEMPERATURE,
        skills: Optional[List[Any]] = None,
    ):
        """Initialize the Skills Agent.

        Args:
            model_name: Name of the language model to use.
            temperature: Temperature setting for the model.
            skills: Optional list of skills to use.
            skills_manager: Optional SkillManager instance.
        """
        self.model_name = model_name
        self.temperature = temperature

        # Initialize skills
        self.skills = skills or SKILLS_REGISTRY
        self.skills_manager = SkillManager(self.skills)

        # Initialize model
        self.llm = create_llm(
            model_name=self.model_name,
            temperature=self.temperature,
            api_key=settings.OPENAI_API_KEY
        )

        # Create agent with skill middleware
        self.agent = self._create_agent()
        self.checkpointer = InMemorySaver()

        logger.info(f"SkillsAgent initialized with {len(self.skills)} skills")

    def _create_agent(self) -> AgentExecutor:
        """Create the agent with SkillMiddleware."""
        # Create the system prompt with skills awareness
        system_prompt = """You are a versatile AI assistant with access to specialized skills.

You can help with various tasks including code analysis, documentation generation,
diagram creation, and more. Each skill provides detailed instructions, tools,
and best practices for specific domains.

Your capabilities:
- Analyze code structure across multiple languages
- Generate documentation with diagrams
- Create visual representations of systems
- Extract and document data models
- Discover and document APIs

You have access to specialized skills for each of these capabilities. Always load
the appropriate skill(s) before starting work to ensure you have the detailed
instructions and tools needed for high-quality results.

When responding:
1. First identify which skills are needed for the user's request
2. Load relevant skills using the load_skill function
3. Follow the skill's guidelines and use its tools
4. Provide a comprehensive response using the loaded knowledge

Remember to load skills even for simple tasks - the skill contains specific
instructions and best practices that ensure quality results.
"""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Create tools list including the skill loading tool
        tools = [load_skill] + self.skills_manager.tools

        # Create agent with SkillMiddleware
        agent = create_agent(
            self.llm,
            tools=tools,
            prompt=prompt,
        )

        # Create executor
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            checkpointer=self.checkpointer,
            max_iterations=20
        )

        # Create executor
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            checkpointer=self.checkpointer,
            max_iterations=AgentConfigDefaults.MAX_ITERATIONS
        )

        return executor

    @handle_agent_error("agent invocation")
    async def invoke(
        self,
        input_data: Dict[str, Any],
        thread_id: Optional[str] = None,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """Invoke the agent with skill middleware support."""
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Apply thread configuration
        if config is None:
            config = RunnableConfig()
        config.update({"configurable": {"thread_id": thread_id}})

        # Invoke the agent
        result = await self.agent.ainvoke(input_data, config)

        return {
            "success": True,
            "content": result.get("output", ""),
            "metadata": {
                "thread_id": thread_id,
                "skills_used": self._extract_skills_from_result(result),
                "total_tokens": result.get("usage", {}).get("total_tokens", 0)
            }
        }

    def _extract_skills_from_result(self, result: Dict[str, Any]) -> List[str]:
        """Extract skill names from the agent result."""
        skills_used = []

        # Look for skill loading indicators in the output
        output = result.get("output", "")
        if SkillMessages.LOADED_SKILL_PATTERN in output:
            # Extract skill names from output
            import re
            matches = re.findall(SkillMessages.LOADED_SKILL_PATTERN, output)
            skills_used.extend(matches)

        return skills_used




# Example usage and standalone execution
async def create_skills_example():
    """Create and run a skills agent example."""
    # Initialize the skills agent
    skills_agent = SkillsAgent()

    # Example usage
    result = await skills_agent.invoke({
        "input": "Analyze the code in a repository and generate documentation with architecture diagrams."
    })

    print(f"Response: {result['content']}")
    print(f"Skills used: {result['metadata']['skills_used']}")

    return result


if __name__ == "__main__":
    import asyncio

    # Run the example
    asyncio.run(create_skills_example())