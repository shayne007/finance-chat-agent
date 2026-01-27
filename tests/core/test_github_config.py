"""Tests for GitHub configuration using TDD approach."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError


class TestGitHubSettings:
    """Test GitHubSettings Pydantic model."""

    def test_settings_loads_from_environment_variables(self):
        """Test GitHubSettings loads from environment variables."""
        from app.core.config import GitHubSettings

        env_vars = {
            "GITHUB_TOKEN": "ghp_test_token",
            "GITHUB_DEFAULT_REPO": "owner/repo",
            "GITHUB_BASE_URL": "https://api.github.com",
            "GITHUB_TIMEOUT": "60",
            "GITHUB_MAX_RETRIES": "5",
            "GITHUB_AGENT_ENABLED": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = GitHubSettings()
            assert settings.token == "ghp_test_token"
            assert settings.default_repo == "owner/repo"
            assert settings.base_url == "https://api.github.com"
            assert settings.timeout == 60
            assert settings.max_retries == 5
            assert settings.enabled is True

    @pytest.mark.parametrize("missing_env_var", [
        "GITHUB_TOKEN",
    ])
    def test_missing_required_fields_raises_validation_error(self, missing_env_var):
        """Test missing required fields raises ValidationError."""
        from app.core.config import GitHubSettings

        env_vars = {
            "GITHUB_DEFAULT_REPO": "owner/repo",
            "GITHUB_BASE_URL": "https://api.github.com",
        }

        # Remove the required field
        env_vars.pop(missing_env_var, None)

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                GitHubSettings()
            # Verify the error mentions the missing field
            assert missing_env_var.lower() in str(exc_info.value).lower()

    @pytest.mark.parametrize("token,should_pass", [
        ("ghp_test_token", True),
        ("github_pat_123", True),
        ("token", True),
        ("", False),  # Empty token should fail
    ])
    def test_token_format_validation(self, token, should_pass):
        """Test token format validation."""
        from app.core.config import GitHubSettings

        env_vars = {
            "GITHUB_TOKEN": token,
            "GITHUB_DEFAULT_REPO": "owner/repo",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            if should_pass:
                settings = GitHubSettings()
                assert settings.token == token
            else:
                with pytest.raises(ValidationError):
                    GitHubSettings()

    @pytest.mark.parametrize("env_vars,expected_values", [
        (
            {
                "GITHUB_TOKEN": "ghp_test",
                "GITHUB_DEFAULT_REPO": "owner/repo",
            },
            {
                "token": "ghp_test",
                "default_repo": "owner/repo",
                "base_url": "https://api.github.com",
                "timeout": 30,
                "max_retries": 3,
                "enabled": False,
            },
        ),
        (
            {
                "GITHUB_TOKEN": "ghp_test",
                "GITHUB_TIMEOUT": "120",
            },
            {
                "token": "ghp_test",
                "default_repo": None,
                "base_url": "https://api.github.com",
                "timeout": 120,
                "max_retries": 3,
                "enabled": False,
            },
        ),
    ])
    def test_default_values_when_optional_fields_not_provided(
        self, env_vars, expected_values
    ):
        """Test default values when optional fields are not provided."""
        from app.core.config import GitHubSettings

        with patch.dict(os.environ, env_vars, clear=True):
            settings = GitHubSettings()
            for key, value in expected_values.items():
                assert getattr(settings, key) == value

    def test_enabled_field_accepts_boolean_string(self):
        """Test enabled field accepts boolean strings."""
        from app.core.config import GitHubSettings

        env_vars = {
            "GITHUB_TOKEN": "ghp_test",
            "GITHUB_AGENT_ENABLED": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = GitHubSettings()
            assert settings.enabled is True

        env_vars["GITHUB_AGENT_ENABLED"] = "false"
        with patch.dict(os.environ, env_vars, clear=True):
            settings = GitHubSettings()
            assert settings.enabled is False

    def test_timeout_converts_to_int(self):
        """Test timeout field converts to integer."""
        from app.core.config import GitHubSettings

        env_vars = {
            "GITHUB_TOKEN": "ghp_test",
            "GITHUB_TIMEOUT": "45",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = GitHubSettings()
            assert settings.timeout == 45
            assert isinstance(settings.timeout, int)

    def test_max_retries_converts_to_int(self):
        """Test max_retries field converts to integer."""
        from app.core.config import GitHubSettings

        env_vars = {
            "GITHUB_TOKEN": "ghp_test",
            "GITHUB_MAX_RETRIES": "10",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = GitHubSettings()
            assert settings.max_retries == 10
            assert isinstance(settings.max_retries, int)
