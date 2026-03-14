"""
Shared API utilities and error handling
"""

from typing import Dict, Any
from fastapi import HTTPException, status


def create_error_response(
    error_type: str,
    detail: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
) -> HTTPException:
    """
    Create a standardized error response.

    Args:
        error_type: Type of error (e.g., "validation_error", "processing_error")
        detail: Error details
        status_code: HTTP status code

    Returns:
        HTTPException with standardized format
    """
    return HTTPException(
        status_code=status_code,
        detail=f"{error_type}: {detail}"
    )


def format_skills_info(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format skills information for API response.

    Args:
        skills: List of skill dictionaries

    Returns:
        Formatted skills info list
    """
    return [
        {
            "name": skill.get("name", ""),
            "description": skill.get("description", ""),
            "category": skill.get("category", ""),
            "token_budget": skill.get("token_budget", 0),
            "tools_available": len(skill.get("tools", [])) if skill.get("tools") else 0
        }
        for skill in skills
    ]


def validate_request_body(request_body: Dict[str, Any], required_fields: List[str]) -> None:
    """
    Validate that required fields are present in request body.

    Args:
        request_body: Request body to validate
        required_fields: List of required field names

    Raises:
        HTTPException: If required fields are missing
    """
    missing_fields = [field for field in required_fields if field not in request_body]
    if missing_fields:
        raise create_error_response(
            "validation_error",
            f"Missing required fields: {', '.join(missing_fields)}",
            status.HTTP_400_BAD_REQUEST
        )