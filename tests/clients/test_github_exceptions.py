"""Tests for GitHub exception hierarchy using TDD approach."""

import pytest


class TestGitHubAPIError:
    """Test GitHubAPIError base exception class."""

    def test_github_api_error_can_be_raised(self):
        """Test GitHubAPIError can be raised and caught."""
        from app.clients.github_client import GitHubAPIError

        with pytest.raises(GitHubAPIError) as exc_info:
            raise GitHubAPIError("Test error message")
        assert "Test error message" in str(exc_info.value)

    def test_github_api_error_with_custom_message(self):
        """Test GitHubAPIError accepts custom message."""
        from app.clients.github_client import GitHubAPIError

        error = GitHubAPIError("Custom error")
        assert str(error) == "Custom error"


class TestGitHubRateLimitError:
    """Test GitHubRateLimitError exception class."""

    def test_rate_limit_error_inherits_from_github_api_error(self):
        """Test GitHubRateLimitError inherits from GitHubAPIError."""
        from app.clients.github_client import GitHubAPIError, GitHubRateLimitError

        assert issubclass(GitHubRateLimitError, GitHubAPIError)

    def test_rate_limit_error_can_be_raised_and_caught(self):
        """Test GitHubRateLimitError can be raised as GitHubAPIError."""
        from app.clients.github_client import GitHubAPIError, GitHubRateLimitError

        with pytest.raises(GitHubAPIError) as exc_info:
            raise GitHubRateLimitError("Rate limit exceeded")
        assert isinstance(exc_info.value, GitHubRateLimitError)
        assert "Rate limit exceeded" in str(exc_info.value)

    def test_rate_limit_error_can_be_caught_directly(self):
        """Test GitHubRateLimitError can be caught directly."""
        from app.clients.github_client import GitHubRateLimitError

        with pytest.raises(GitHubRateLimitError) as exc_info:
            raise GitHubRateLimitError("Rate limit exceeded")
        assert "Rate limit exceeded" in str(exc_info.value)


class TestGitHubAuthenticationError:
    """Test GitHubAuthenticationError exception class."""

    def test_auth_error_inherits_from_github_api_error(self):
        """Test GitHubAuthenticationError inherits from GitHubAPIError."""
        from app.clients.github_client import GitHubAPIError, GitHubAuthenticationError

        assert issubclass(GitHubAuthenticationError, GitHubAPIError)

    def test_auth_error_can_be_raised_and_caught(self):
        """Test GitHubAuthenticationError can be raised as GitHubAPIError."""
        from app.clients.github_client import GitHubAPIError, GitHubAuthenticationError

        with pytest.raises(GitHubAPIError) as exc_info:
            raise GitHubAuthenticationError("Invalid token")
        assert isinstance(exc_info.value, GitHubAuthenticationError)
        assert "Invalid token" in str(exc_info.value)

    def test_auth_error_can_be_caught_directly(self):
        """Test GitHubAuthenticationError can be caught directly."""
        from app.clients.github_client import GitHubAuthenticationError

        with pytest.raises(GitHubAuthenticationError) as exc_info:
            raise GitHubAuthenticationError("Authentication failed")
        assert "Authentication failed" in str(exc_info.value)


class TestGitHubNotFoundError:
    """Test GitHubNotFoundError exception class."""

    def test_not_found_error_inherits_from_github_api_error(self):
        """Test GitHubNotFoundError inherits from GitHubAPIError."""
        from app.clients.github_client import GitHubAPIError, GitHubNotFoundError

        assert issubclass(GitHubNotFoundError, GitHubAPIError)

    def test_not_found_error_can_be_raised_and_caught(self):
        """Test GitHubNotFoundError can be raised as GitHubAPIError."""
        from app.clients.github_client import GitHubAPIError, GitHubNotFoundError

        with pytest.raises(GitHubAPIError) as exc_info:
            raise GitHubNotFoundError("Resource not found")
        assert isinstance(exc_info.value, GitHubNotFoundError)
        assert "Resource not found" in str(exc_info.value)

    def test_not_found_error_can_be_caught_directly(self):
        """Test GitHubNotFoundError can be caught directly."""
        from app.clients.github_client import GitHubNotFoundError

        with pytest.raises(GitHubNotFoundError) as exc_info:
            raise GitHubNotFoundError("Issue not found")
        assert "Issue not found" in str(exc_info.value)


class TestGitHubValidationError:
    """Test GitHubValidationError exception class."""

    def test_validation_error_inherits_from_github_api_error(self):
        """Test GitHubValidationError inherits from GitHubAPIError."""
        from app.clients.github_client import GitHubAPIError, GitHubValidationError

        assert issubclass(GitHubValidationError, GitHubAPIError)

    def test_validation_error_can_be_raised_and_caught(self):
        """Test GitHubValidationError can be raised as GitHubAPIError."""
        from app.clients.github_client import GitHubAPIError, GitHubValidationError

        with pytest.raises(GitHubAPIError) as exc_info:
            raise GitHubValidationError("Invalid input")
        assert isinstance(exc_info.value, GitHubValidationError)
        assert "Invalid input" in str(exc_info.value)

    def test_validation_error_can_be_caught_directly(self):
        """Test GitHubValidationError can be caught directly."""
        from app.clients.github_client import GitHubValidationError

        with pytest.raises(GitHubValidationError) as exc_info:
            raise GitHubValidationError("Validation failed")
        assert "Validation failed" in str(exc_info.value)


class TestAllExceptionsCanBeCaughtAsGitHubAPIError:
    """Test all custom exceptions can be caught as GitHubAPIError."""

    @pytest.mark.parametrize("exception_class,exception_message", [
        ("GitHubRateLimitError", "Rate limit exceeded"),
        ("GitHubAuthenticationError", "Bad credentials"),
        ("GitHubNotFoundError", "Not found"),
        ("GitHubValidationError", "Invalid data"),
    ])
    def test_all_exceptions_caught_as_base(self, exception_class, exception_message):
        """Test all custom exceptions can be caught as GitHubAPIError."""
        from app.clients.github_client import GitHubAPIError

        # Get the exception class dynamically
        exception_classes = {
            "GitHubRateLimitError": "GitHubRateLimitError",
            "GitHubAuthenticationError": "GitHubAuthenticationError",
            "GitHubNotFoundError": "GitHubNotFoundError",
            "GitHubValidationError": "GitHubValidationError",
        }

        # Import based on the class name
        if exception_class == "GitHubRateLimitError":
            from app.clients.github_client import GitHubRateLimitError as ExcClass
        elif exception_class == "GitHubAuthenticationError":
            from app.clients.github_client import GitHubAuthenticationError as ExcClass
        elif exception_class == "GitHubNotFoundError":
            from app.clients.github_client import GitHubNotFoundError as ExcClass
        else:
            from app.clients.github_client import GitHubValidationError as ExcClass

        with pytest.raises(GitHubAPIError) as exc_info:
            raise ExcClass(exception_message)

        assert str(exc_info.value) == exception_message
        assert type(exc_info.value).__name__ == exception_class
