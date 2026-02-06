"""
Celery tasks for repository documentation generation.
"""

import os
import json
from datetime import datetime
from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.conversation import Conversation, Message
from app.services.repository_service import RepositoryService
from app.core.config import github_settings
from app.clients.github_client import GitHubClient


@celery_app.task
def generate_repository_docs_task(
    message_id: str,
    user_id: str,
    repo_url: str,
    branch: str = "main",
    output_dir: str = None,
    max_files: int = 100
) -> str:
    """
    Generate markdown documentation from a GitHub repository.

    Args:
        message_id: ID of the original message
        user_id: User ID requesting the documentation
        repo_url: GitHub repository URL
        branch: Branch to clone (default: main)
        output_dir: Output directory for documentation
        max_files: Maximum number of code files to analyze

    Returns:
        Task status: 'success', 'failed', or 'pending'
    """
    db: Session = SessionLocal()
    try:
        # Get the original message
        user_msg = db.query(Message).filter(Message.id == message_id).first()
        if not user_msg:
            return "not_found"

        # Check conversation ownership
        conv = db.query(Conversation).filter(
            Conversation.id == user_msg.conversation_id,
            Conversation.user_id == user_id
        ).first()
        if not conv:
            meta = json.loads(user_msg.meta) if user_msg.meta else {}
            meta.update({
                "status": "failed",
                "error": "unauthorized",
                "timestamp": datetime.now().isoformat()
            })
            user_msg.meta = json.dumps(meta)
            db.add(user_msg)
            db.commit()
            return "unauthorized"

        # Initialize repository service
        github_client = GitHubClient(
            token=github_settings.token,
            timeout=github_settings.timeout,
            max_retries=github_settings.max_retries
        )

        repository_service = RepositoryService()
        repository_service.cloner.github_client = github_client

        # Update message status
        meta = json.loads(user_msg.meta) if user_msg.meta else {}
        meta.update({
            "status": "processing",
            "repo_url": repo_url,
            "branch": branch,
            "started_at": datetime.now().isoformat()
        })
        user_msg.meta = json.dumps(meta)
        db.add(user_msg)
        db.commit()

        # Generate documentation
        result = repository_service.transform_repository_to_markdown(
            url=repo_url,
            branch=branch,
            output_dir=output_dir,
            max_files=max_files
        )

        # Update message with result
        meta.update({
            "status": "success" if not result.errors else "partial_success",
            "completed_at": datetime.now().isoformat(),
            "output_dir": result.output_dir,
            "files_created": len(result.files_created),
            "summary": result.summary,
            "errors": result.errors
        })
        user_msg.meta = json.dumps(meta)
        user_msg.content = json.dumps({
            "output_dir": result.output_dir,
            "files_created": result.files_created,
            "summary": result.summary
        })
        db.add(user_msg)
        db.commit()

        return "success"

    except Exception as e:
        # Update message with error
        meta = json.loads(user_msg.meta) if user_msg.meta else {}
        meta.update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        user_msg.meta = json.dumps(meta)
        db.add(user_msg)
        db.commit()
        return "failed"

    finally:
        db.close()


@celery_app.task
def check_repo_size_task(repo_url: str) -> dict:
    """
    Check the size of a GitHub repository to determine if it can be processed.

    Args:
        repo_url: GitHub repository URL

    Returns:
        Dictionary with size information
    """
    try:
        # Extract owner and repo from URL
        if not repo_url.startswith(('https://github.com/', 'http://github.com/')):
            return {
                "valid": False,
                "error": "Invalid GitHub URL"
            }

        repo_parts = repo_url.replace('https://github.com/', '').replace('http://github.com/', '').split('/')
        if len(repo_parts) < 2:
            return {
                "valid": False,
                "error": "Invalid repository URL format"
            }

        owner, repo = repo_parts[0], repo_parts[1].replace('.git', '')

        # Note: In a real implementation, you would use the GitHub API
        # to get repository information. For now, we'll return a mock response.

        return {
            "valid": True,
            "owner": owner,
            "repo": repo,
            "size_mb": 25,  # Mock size
            "can_process": True,
            "message": "Repository size is within limits"
        }

    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }


@celery_app.task
def cleanup_temp_directories_task() -> dict:
    """
    Clean up temporary directories used for repository cloning.

    Returns:
        Dictionary with cleanup statistics
    """
    import shutil
    from pathlib import Path
    from app.services.repository_service import RepositoryCloner

    cleaned_count = 0
    error_count = 0
    cleaned_dirs = []

    try:
        # Get temporary directory
        temp_dir = Path(tempfile.gettempdir()) / "repo_cloning"

        if temp_dir.exists():
            # Find directories older than 24 hours
            for repo_dir in temp_dir.iterdir():
                if repo_dir.is_dir():
                    try:
                        # Check if directory is older than 24 hours
                        file_age = datetime.now().timestamp() - repo_dir.stat().st_mtime
                        if file_age > 86400:  # 24 hours in seconds
                            shutil.rmtree(repo_dir)
                            cleaned_count += 1
                            cleaned_dirs.append(str(repo_dir))
                    except Exception:
                        error_count += 1

        return {
            "success": True,
            "cleaned_directories": cleaned_count,
            "errors": error_count,
            "cleaned_paths": cleaned_dirs,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }