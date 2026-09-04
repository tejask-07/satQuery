"""
Phase 9: Canonical Repository Path Resolution.

Ensures that evaluation modules, tests, CLI runners, and API routes
resolve benchmark manifest, data caches, reference rasters, and reports
to their single canonical location inside `backend/` regardless of
whether the current working directory is the repository root or `backend/`.
"""

from __future__ import annotations

from pathlib import Path


def get_backend_dir() -> Path:
    """Returns the absolute Path to the backend directory."""
    # This file is at backend/app/evaluation/paths.py
    return Path(__file__).resolve().parent.parent.parent


def get_repo_root() -> Path:
    """Returns the absolute Path to the repository root."""
    return get_backend_dir().parent


def resolve_repo_path(rel_or_abs_path: str | Path) -> Path:
    """
    Resolves a relative or absolute path against the canonical backend structure.

    Canonical locations:
    - backend/data/benchmark/
    - backend/data/cache/
    - backend/reports/

    Resolution order:
    1. If already an absolute Path and exists -> return it.
    2. If relative to backend/ exists -> return backend / rel_path.
    3. If relative to repo root exists -> return repo_root / rel_path.
    4. If relative to cwd exists -> return (Path.cwd() / rel_path).resolve().
    5. Default to backend / rel_path (canonical target).
    """
    p = Path(rel_or_abs_path)
    if p.is_absolute():
        return p

    backend = get_backend_dir()
    repo_root = get_repo_root()

    # If the path starts with "backend/", try repo_root / path
    parts = p.parts
    if parts and parts[0] == "backend":
        cand = repo_root / p
        if cand.exists():
            return cand
        # Also check without the "backend" prefix inside backend
        sub_p = Path(*parts[1:])
        cand_sub = backend / sub_p
        if cand_sub.exists():
            return cand_sub

    # Check directly inside backend
    cand_backend = backend / p
    if cand_backend.exists():
        return cand_backend

    # Check relative to repo root
    cand_root = repo_root / p
    if cand_root.exists():
        return cand_root

    # Check relative to cwd
    cand_cwd = (Path.cwd() / p).resolve()
    if cand_cwd.exists():
        return cand_cwd

    # Fallback to backend/relative path as canonical location
    return cand_backend


def get_manifest_path() -> Path:
    """Returns canonical path to benchmark manifest.json."""
    return resolve_repo_path("data/benchmark/manifest.json")


def get_reports_dir() -> Path:
    """Returns canonical path to backend/reports directory."""
    return resolve_repo_path("reports")


def get_cache_dir() -> Path:
    """Returns canonical path to backend/data/cache directory."""
    return resolve_repo_path("data/cache")


def get_aligned_dir() -> Path:
    """Returns canonical path to backend/data/benchmark/aligned directory."""
    return resolve_repo_path("data/benchmark/aligned")
