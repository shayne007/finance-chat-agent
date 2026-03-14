"""
Shared error handling utilities
"""

from fastapi import HTTPException, status
from typing import Type


def create_http_exception(
    error_class: Type[HTTPException],
    detail: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
) -> HTTPException:
    """Create a standardized HTTP exception."""
    return error_class(
        status_code=status_code,
        detail=detail
    )


def handle_agent_error(operation: str) -> callable:
    """Decorator for handling agent-related errors."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                raise create_http_exception(
                    HTTPException,
                    f"{operation} failed: {str(e)}"
                )
        return wrapper
    return decorator