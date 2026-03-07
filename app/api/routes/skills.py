"""Skills-based API routes for the finance chat agent."""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
from uuid import UUID

from app.config.agents import AgentConfig, AgentInfo
from app.agents.skills_agent import SkillsAgent


router = APIRouter()


@router.get("/info")
async def get_skills_info() -> Dict[str, Any]:
    """Get information about available skills and agent capabilities.

    Returns:
        Dictionary with skills information and agent capabilities
    """
    # Initialize skills
    skills_manager = AgentConfig.create_finance_agent(enable_skills=True).skills_manager

    # Get agent info
    finance_agent = AgentConfig.create_finance_agent(enable_skills=True)
    finance_info = AgentConfig.get_agent_info(finance_agent)

    skills_info = []
    for skill in skills_manager.skills:
        skills_info.append({
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "token_budget": skill.token_budget,
            "tools_available": len(skill.tools) if skill.tools else 0
        })

    return {
        "skills_count": len(skills_manager.skills),
        "skills": skills_info,
        "agent_capabilities": finance_info,
        "supported_tasks": [
            "Code analysis",
            "Documentation generation",
            "Diagram creation",
            "Data modeling",
            "API discovery",
            "Architecture analysis"
        ]
    }


@router.post("/analyze")
async def analyze_code_with_skills(
    request: Dict[str, Any],
    enable_skills: bool = True,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Analyze code using skills-based agent.

    Args:
        request: Input request containing the task description and code/data
        enable_skills: Whether to enable skills support
        model: Model to use for analysis
        temperature: Temperature setting for the model

    Returns:
        Analysis result with skill usage information
    """
    try:
        # Create skills agent
        skills_agent = AgentConfig.create_skills_only_agent(
            model_name=model,
            temperature=temperature
        )

        # Get thread ID from request or create new one
        thread_id = request.get("thread_id") or f"skills-{UUID().hex}"

        # Prepare input
        input_data = {
            "input": request.get("input", ""),
            **{k: v for k, v in request.items() if k != "input" and k != "thread_id"}
        }

        # Run analysis
        result = await skills_agent.invoke(input_data, thread_id=thread_id)

        # Add metadata
        result["metadata"].update({
            "request_id": thread_id,
            "model_used": model,
            "skills_enabled": enable_skills
        })

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/generate-documentation")
async def generate_documentation_with_skills(
    request: Dict[str, Any],
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """Generate documentation using skills-based agent.

    Args:
        request: Request containing code repository path and requirements
        model: Model to use for generation

    Returns:
        Generated documentation with skill usage info
    """
    try:
        # Create skills agent
        skills_agent = AgentConfig.create_skills_only_agent(
            model_name=model
        )

        # Prepare input
        input_data = {
            "input": f"""Generate comprehensive documentation for the following requirements:

Repository: {request.get('repo_path', 'Not specified')}
Format: {request.get('format', 'markdown')}
Requirements: {request.get('requirements', 'Standard documentation')}

Additional context:
{request.get('context', '')}
""",
            **{k: v for k, v in request.items()
               if k not in ['repo_path', 'format', 'requirements', 'context']}
        }

        # Generate documentation
        result = await skills_agent.invoke(
            input_data,
            thread_id=f"doc-gen-{UUID().hex}"
        )

        # Add metadata
        result["metadata"].update({
            "generation_type": "documentation",
            "repo_path": request.get('repo_path'),
            "format": request.get('format')
        })

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Documentation generation failed: {str(e)}"
        )


@router.post("/create-architecture-diagram")
async def create_architecture_diagram(
    request: Dict[str, Any],
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """Create architecture diagrams using skills.

    Args:
        request: Request containing architecture description and requirements
        model: Model to use for diagram generation

    Returns:
        Generated diagram with skill usage info
    """
    try:
        # Create skills agent
        skills_agent = AgentConfig.create_skills_only_agent(
            model_name=model
        )

        # Prepare input
        input_data = {
            "input": f"""Create architecture diagrams based on the following description:

Description: {request.get('description', '')}
Diagram Type: {request.get('diagram_type', 'architecture')}
Requirements: {request.get('requirements', 'Standard system architecture')}

Additional details:
{request.get('details', '')}
""",
            **{k: v for k, v in request.items()
               if k not in ['description', 'diagram_type', 'requirements', 'details']}
        }

        # Generate diagram
        result = await skills_agent.invoke(
            input_data,
            thread_id=f"diagram-{UUID().hex}"
        )

        # Add metadata
        result["metadata"].update({
            "diagram_type": request.get('diagram_type', 'architecture')
        })

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagram generation failed: {str(e)}"
        )


@router.get("/usage/{thread_id}")
async def get_skill_usage(thread_id: str) -> Dict[str, Any]:
    """Get skill usage information for a specific thread.

    Args:
        thread_id: Thread ID to retrieve usage for

    Returns:
        Skill usage statistics for the thread
    """
    try:
        # This would typically retrieve from a monitoring system
        # For now, return a placeholder
        return {
            "thread_id": thread_id,
            "skills_used": [],  # Would be populated from actual tracking
            "total_tokens": 0,
            "messages_count": 0,
            "last_activity": None
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve usage: {str(e)}"
        )