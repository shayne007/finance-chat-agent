"""Tests for SkillMiddleware implementation."""

import pytest
from unittest.mock import Mock, AsyncMock
from app.skills.middleware import SkillMiddleware
from app.skills.types import Skill
from app.skills.registry import load_skill


@pytest.fixture
def mock_skills():
    """Mock skills for testing."""
    return [
        Skill(
            name="code_analysis",
            description="Analyze code structure",
            content="# Detailed code analysis instructions",
            category="analysis",
            token_budget=2000
        ),
        Skill(
            name="documentation",
            description="Generate documentation",
            content="# Documentation generation instructions",
            category="generation",
            token_budget=2500
        )
    ]


@pytest.fixture
def skill_middleware(mock_skills):
    """SkillMiddleware fixture."""
    return SkillMiddleware(mock_skills)


def test_skill_middleware_initialization(skill_middleware, mock_skills):
    """Test SkillMiddleware initialization."""
    assert len(skill_middleware.skills) == 2
    assert skill_middleware.tools == [load_skill]


def test_build_level1_prompt(skill_middleware, mock_skills):
    """Test Level 1 prompt building."""
    prompt = skill_middleware._build_level1_prompt()

    assert "**Analysis Skills:**" in prompt
    assert "**Code Analysis:**: Analyze code structure" in prompt
    assert "**Generation Skills:**" in prompt
    assert "**Documentation:**: Generate documentation" in prompt


@pytest.mark.asyncio
async def test_skill_middleware_wrap_model_call(skill_middleware):
    """Test model call wrapping."""
    # Mock handler
    mock_handler = AsyncMock()
    mock_handler.return_value = Mock()
    mock_handler.return_value.content = "Test response"

    # Mock request with system message
    from langchain.agents.middleware import ModelRequest
    from langchain.messages import SystemMessage

    request = ModelRequest(
        system_message=SystemMessage(
            content="You are a helpful assistant."
        ),
        messages=[]
    )

    # Mock config
    mock_config = Mock()

    # Wrap model call
    response = skill_middleware.wrap_model_call(request, mock_handler, mock_config)

    # Verify handler was called
    mock_handler.assert_called_once()

    # Verify system message was enhanced
    assert "Available Skills" in request.system_message.content
    assert "Code Analysis:**: Analyze code structure" in request.system_message.content


def test_load_skill_functionality():
    """Test load_skill function."""
    from app.skills.registry import SKILLS_REGISTRY

    # Test loading existing skill
    result = load_skill("code_analysis")
    assert "Loaded Skill: code_analysis" in result
    assert "Category: analysis" in result
    assert "Token Budget: 2000" in result

    # Test loading non-existent skill
    result = load_skill("nonexistent")
    assert "Skill 'nonexistent' not found" in result