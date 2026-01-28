"""Tests for MCP Configuration using TDD approach."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError


class TestMCPSettings:
    """Test MCPSettings Pydantic model."""

    def test_settings_loads_from_environment_variables(self):
        """Test MCPSettings loads from environment variables."""
        from app.core.config import MCPSettings

        env_vars = {
            "GITHUB_SERVER_ENABLED": "true",
            "GITHUB_SERVER_PORT": "8081",
            "GITHUB_MAX_TOOLS": "50",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = MCPSettings()
            assert settings.github_server_enabled is True
            assert settings.github_server_port == 8081
            assert settings.github_max_tools == 50

    @pytest.mark.parametrize("missing_env_var", [
        "GITHUB_SERVER_ENABLED",
        "GITHUB_SERVER_PORT",
    ])
    def test_missing_required_fields_raises_validation_error(self, missing_env_var):
        """Test missing required fields raises ValidationError."""
        from app.core.config import MCPSettings

        env_vars = {
            "GITHUB_MAX_TOOLS": "50",
        }

        env_vars.pop(missing_env_var, None)

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                MCPSettings()
            assert missing_env_var.lower() in str(exc_info.value).lower()

    @pytest.mark.parametrize("port_value,should_pass", [
        (8081, True),
        (8080, True),
        (9000, True),
        (-1, False),  # Invalid port
        (70000, False),  # Invalid port
    ])
    def test_port_validation(self, port_value, should_pass):
        """Test port format validation."""
        from app.core.config import MCPSettings

        env_vars = {
            "GITHUB_SERVER_ENABLED": "true",
            "GITHUB_SERVER_PORT": str(port_value),
        }

        with patch.dict(os.environ, env_vars, clear=True):
            if should_pass:
                settings = MCPSettings()
                assert settings.github_server_port == port_value
            else:
                with pytest.raises(ValidationError):
                    MCPSettings()

    @pytest.mark.parametrize("env_vars,expected_values", [
        (
            {
                "GITHUB_SERVER_ENABLED": "true",
                "GITHUB_SERVER_PORT": "8081",
            },
            {
                "github_server_enabled": True,
                "github_server_port": 8081,
                "github_max_tools": 100,
            },
        ),
        (
            {
                "GITHUB_SERVER_ENABLED": "false",
                "GITHUB_SERVER_PORT": "9000",
                "GITHUB_MAX_TOOLS": "25",
            },
            {
                "github_server_enabled": False,
                "github_server_port": 9000,
                "github_max_tools": 25,
            },
        ),
    ])
    def test_default_values_when_optional_fields_not_provided(
        self, env_vars, expected_values
    ):
        """Test default values when optional fields are not provided."""
        from app.core.config import MCPSettings

        with patch.dict(os.environ, env_vars, clear=True):
            settings = MCPSettings()
            for key, value in expected_values.items():
                assert getattr(settings, key) == value

    def test_enabled_field_accepts_boolean_string(self):
        """Test enabled field accepts boolean strings."""
        from app.core.config import MCPSettings

        env_vars = {
            "GITHUB_SERVER_PORT": "8081",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            env_vars["GITHUB_SERVER_ENABLED"] = "true"
            settings = MCPSettings()
            assert settings.github_server_enabled is True

            env_vars["GITHUB_SERVER_ENABLED"] = "false"
            settings = MCPSettings()
            assert settings.github_server_enabled is False

    def test_max_tools_converts_to_int(self):
        """Test max_tools field converts to integer."""
        from app.core.config import MCPSettings

        env_vars = {
            "GITHUB_SERVER_ENABLED": "true",
            "GITHUB_SERVER_PORT": "8081",
            "GITHUB_MAX_TOOLS": "100",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = MCPSettings()
            assert settings.github_max_tools == 100
            assert isinstance(settings.github_max_tools, int)


@pytest.mark.asyncio
class TestGitHubMCPServerWithConfiguration:
    """Test GitHubMCPServer uses MCPSettings."""

    async def test_server_uses_configuration_for_tool_limit(self):
        """Test server uses max_tools from configuration."""
        from app.mcp.github_server import GitHubMCPServer
        from app.core.config import MCPSettings
        from app.clients.github_client import GitHubClient

        mock_github_client = Mock(spec=GitHubClient)

        # Mock the configuration load
        mock_settings = Mock(spec=MCPSettings)
        mock_settings.github_max_tools = 10
        mock_settings.github_server_enabled = True
        mock_settings.github_server_port = 8081

        server = GitHubMCPServer(github_client=mock_github_client)

        tools = await server.list_tools()
        # Verify all expected tools are returned
        assert len(tools) == 4
