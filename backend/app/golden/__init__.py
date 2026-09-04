"""
Phase 10: Golden End-to-End Validation Suite Package.
"""

from app.golden.manifest import GoldenQuery, load_golden_manifest
from app.golden.validators import (
    validate_golden_result,
    GoldenValidationResult,
    GoldenValidationError,
)
from app.golden.runner import run_golden_suite, generate_golden_report

__all__ = [
    "GoldenQuery",
    "load_golden_manifest",
    "validate_golden_result",
    "GoldenValidationResult",
    "GoldenValidationError",
    "run_golden_suite",
    "generate_golden_report",
]
