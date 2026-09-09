"""
Validation schemas and error codes for remote-sensing workflows.

Defines the standard validation result format and error codes used across
all SatQuery backend ingestion, upload, query planning, and execution layers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ValidationErrorCode:
    """Standardized validation error codes across SatQuery workflows."""
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    INVALID_IMAGE = "INVALID_IMAGE"
    MISSING_IMAGE = "MISSING_IMAGE"
    WRONG_IMAGE_COUNT = "WRONG_IMAGE_COUNT"
    INCOMPATIBLE_IMAGE_PAIR = "INCOMPATIBLE_IMAGE_PAIR"
    INCOMPATIBLE_QUERY_INPUT = "INCOMPATIBLE_QUERY_INPUT"
    MISSING_METADATA = "MISSING_METADATA"
    SECURITY_ERROR = "SECURITY_ERROR"


class ValidationResult(BaseModel):
    """
    Standard structured validation result.

    Attributes:
        valid: True if input passes validation; False otherwise.
        error_code: Standard machine-readable error code if invalid.
        message: Human-readable actionable description.
        details: Additional context (e.g. received format, supported formats, bounds).
    """
    valid: bool
    error_code: Optional[str] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation matching API contract."""
        data: Dict[str, Any] = {
            "valid": self.valid,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details or {},
        }
        return data

    def to_http_detail(self) -> Dict[str, Any]:
        """
        Return structured dictionary suitable for FastAPI HTTPException payload.
        Includes 'detail' string key to ensure full backward compatibility with tests
        and clients inspecting either dict fields or response['detail'].
        """
        data = self.to_dict()
        data["detail"] = self.message or ""
        return data


class ValidationError(ValueError):
    """Exception raised when backend input validation fails."""

    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__(result.message or "Validation failed")
