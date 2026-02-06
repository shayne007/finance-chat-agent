"""
Agent for handling repository documentation generation requests.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from app.core.config import repository_service_settings, github_settings
from app.tasks.repository_tasks import generate_repository_docs_task, check_repo_size_task
from app.agents.base_agent import BaseAgent


logger = logging.getLogger(__name__)


@dataclass
class RepositoryDocRequest:
    """Request for repository documentation generation."""
    repo_url: str
    branch: str = "main"
    output_dir: Optional[str] = None
    max_files: int = 100
    user_id: Optional[str] = None
    message_id: Optional[str] = None


@dataclass
class RepositoryDocResponse:
    """Response for repository documentation generation."""
    status: str
    message: str
    output_dir: Optional[str] = None
    files_created: Optional[List[str]] = None
    summary: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    task_id: Optional[str] = None


class RepositoryDocAgent(BaseAgent):
    """Agent for generating documentation from GitHub repositories."""

    def __init__(self):
        super().__init__("repository_doc")
        self.settings = repository_service_settings
        self.github_settings = github_settings

    def handle_request(self, request: RepositoryDocRequest) -> RepositoryDocResponse:
        """Handle repository documentation generation request."""
        try:
            # Validate repository URL
            if not self._validate_repo_url(request.repo_url):
                return RepositoryDocResponse(
                    status="error",
                    message="Invalid GitHub repository URL"
                )

            # Check if repository service is enabled
            if not self.settings.enabled:
                return RepositoryDocResponse(
                    status="error",
                    message="Repository documentation generation is disabled"
                )

            # Set default output directory if not provided
            if not request.output_dir:
                request.output_dir = self.settings.output_dir

            # Check repository size (if needed)
            if self.settings.max_repo_size_mb > 0:
                size_check = check_repo_size_task.delay(request.repo_url)
                size_result = size_check.get(timeout=30)

                if not size_result["valid"]:
                    return RepositoryDocResponse(
                        status="error",
                        message=f"Repository size check failed: {size_result.get('error', 'Unknown error')}"
                    )

                if size_result.get("size_mb", 0) > self.settings.max_repo_size_mb:
                    return RepositoryDocResponse(
                        status="error",
                        message=f"Repository too large: {size_result.get('size_mb')}MB > {self.settings.max_repo_size_mb}MB"
                    )

            # Generate documentation
            task_result = generate_repository_docs_task.delay(
                message_id=request.message_id or "",
                user_id=request.user_id or "",
                repo_url=request.repo_url,
                branch=request.branch,
                output_dir=request.output_dir,
                max_files=min(request.max_files, self.settings.max_files)
            )

            # Return response with task information
            return RepositoryDocResponse(
                status="processing",
                message="Documentation generation started",
                task_id=task_result.id
            )

        except Exception as e:
            logger.error(f"Error generating repository documentation: {str(e)}")
            return RepositoryDocResponse(
                status="error",
                message=f"Failed to start documentation generation: {str(e)}"
            )

    def get_task_status(self, task_id: str) -> RepositoryDocResponse:
        """Get the status of a documentation generation task."""
        try:
            from app.core.celery_app import celery_app

            task = celery_app.AsyncResult(task_id)

            if task.state == "PENDING":
                return RepositoryDocResponse(
                    status="pending",
                    message="Task is waiting to be processed",
                    task_id=task_id
                )
            elif task.state == "PROGRESS":
                return RepositoryDocResponse(
                    status="processing",
                    message="Task is being processed",
                    task_id=task_id
                )
            elif task.state == "SUCCESS":
                # Parse result from successful task
                result = task.result
                if isinstance(result, str) and result == "success":
                    return RepositoryDocResponse(
                        status="success",
                        message="Documentation generated successfully",
                        task_id=task_id
                    )
                else:
                    return RepositoryDocResponse(
                        status="failed",
                        message="Task completed but with errors",
                        task_id=task_id
                    )
            elif task.state == "FAILURE":
                return RepositoryDocResponse(
                    status="failed",
                    message=f"Task failed: {str(task.info)}",
                    task_id=task_id
                )
            else:
                return RepositoryDocResponse(
                    status="unknown",
                    message=f"Unknown task state: {task.state}",
                    task_id=task_id
                )

        except Exception as e:
            logger.error(f"Error checking task status: {str(e)}")
            return RepositoryDocResponse(
                status="error",
                message=f"Failed to check task status: {str(e)}",
                task_id=task_id
            )

    def _validate_repo_url(self, repo_url: str) -> bool:
        """Validate GitHub repository URL."""
        if not repo_url:
            return False

        # Check if URL starts with GitHub URL
        if not repo_url.startswith(('https://github.com/', 'http://github.com/')):
            return False

        # Extract repo parts
        try:
            repo_parts = repo_url.replace('https://github.com/', '').replace('http://github.com/', '').split('/')
            if len(repo_parts) < 2:
                return False

            owner, repo = repo_parts[0], repo_parts[1].replace('.git', '')

            # Validate owner and repo names
            if not owner or not repo:
                return False

            # Check for invalid characters
            invalid_chars = [' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\t']
            if any(char in owner or char in repo for char in invalid_chars):
                return False

            return True

        except Exception:
            return False

    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return self.settings.supported_languages

    def get_settings(self) -> Dict[str, Any]:
        """Get repository service settings."""
        return {
            "enabled": self.settings.enabled,
            "output_dir": self.settings.output_dir,
            "max_files": self.settings.max_files,
            "default_branch": self.settings.default_branch,
            "enable_diagrams": self.settings.enable_diagrams,
            "max_repo_size_mb": self.settings.max_repo_size_mb,
            "supported_languages": self.settings.supported_languages
        }

    def generate_sample_request(self) -> RepositoryDocRequest:
        """Generate a sample request for testing."""
        return RepositoryDocRequest(
            repo_url="https://github.com/example/repo",
            branch="main",
            output_dir="./codebase-to-knowledge-docs",
            max_files=50
        )

    def validate_request_format(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Validate request data format and return error message if invalid."""
        required_fields = ["repo_url"]

        for field in required_fields:
            if field not in request_data:
                return f"Missing required field: {field}"

        # Validate repo URL format
        repo_url = request_data.get("repo_url")
        if not self._validate_repo_url(repo_url):
            return "Invalid repository URL format"

        # Validate branch if provided
        branch = request_data.get("branch")
        if branch and not isinstance(branch, str):
            return "Branch must be a string"

        # Validate max_files if provided
        max_files = request_data.get("max_files")
        if max_files is not None:
            try:
                max_files = int(max_files)
                if max_files <= 0:
                    return "Max files must be greater than 0"
            except ValueError:
                return "Max files must be an integer"

        # Validate output_dir if provided
        output_dir = request_data.get("output_dir")
        if output_dir and not isinstance(output_dir, str):
            return "Output directory must be a string"

        return None


# Initialize agent
repository_doc_agent = RepositoryDocAgent()