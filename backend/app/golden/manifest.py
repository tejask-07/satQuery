"""
Phase 10: Golden Query Manifest Loader and Validator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.evaluation.paths import resolve_repo_path


@dataclass
class GoldenQuery:
    """Represents a single validated golden query definition."""
    id: str
    name: str
    query: str
    aoi: List[float]
    expected_intent: str
    expected_task: str
    expected_domain: Optional[str]
    expected_temporal_mode: str
    expected_min_observations: int
    expected_primary_indicator: Optional[str] = None
    expected_source: Optional[str] = None
    expected_destination: Optional[str] = None
    expected_time_start: Optional[str] = None
    expected_time_end: Optional[str] = None
    expected_properties: List[str] = field(default_factory=list)
    category: str = "core_deterministic"
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_golden_manifest_path() -> Path:
    """Return canonical path to backend/data/golden/manifest.json."""
    return resolve_repo_path("data/golden/manifest.json")


def load_golden_manifest(manifest_path: Optional[Path | str] = None) -> List[GoldenQuery]:
    """
    Loads and validates the golden query manifest.

    Raises:
        FileNotFoundError: If the manifest JSON does not exist.
        ValueError: If manifest schema or query definitions are invalid.
    """
    path = Path(manifest_path) if manifest_path else get_golden_manifest_path()
    path = resolve_repo_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Golden query manifest not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Golden manifest root must be a JSON object.")

    raw_queries = data.get("queries")
    if not isinstance(raw_queries, list) or len(raw_queries) == 0:
        raise ValueError("Golden manifest must contain a non-empty 'queries' list.")

    golden_queries: List[GoldenQuery] = []
    seen_ids: set[str] = set()

    for idx, q_dict in enumerate(raw_queries):
        qid = q_dict.get("id")
        if not qid or not isinstance(qid, str):
            raise ValueError(f"Query at index {idx} has invalid or missing 'id'.")
        if qid in seen_ids:
            raise ValueError(f"Duplicate query id '{qid}' detected at index {idx}.")
        seen_ids.add(qid)

        query_str = q_dict.get("query")
        if not query_str or not isinstance(query_str, str):
            raise ValueError(f"Query '{qid}' has missing or empty 'query' text.")

        aoi = q_dict.get("aoi")
        if not isinstance(aoi, list) or len(aoi) != 4 or not all(isinstance(x, (int, float)) for x in aoi):
            raise ValueError(f"Query '{qid}' has invalid 'aoi'; must be [minLon, minLat, maxLon, maxLat].")

        expected_intent = q_dict.get("expected_intent")
        if not expected_intent:
            raise ValueError(f"Query '{qid}' missing 'expected_intent'.")

        expected_task = q_dict.get("expected_task")
        if not expected_task:
            raise ValueError(f"Query '{qid}' missing 'expected_task'.")

        expected_temporal_mode = q_dict.get("expected_temporal_mode", "bi_temporal")
        expected_min_obs = int(q_dict.get("expected_min_observations", 2))

        golden_queries.append(
            GoldenQuery(
                id=qid,
                name=q_dict.get("name", qid),
                query=query_str,
                aoi=aoi,
                expected_intent=expected_intent,
                expected_task=expected_task,
                expected_domain=q_dict.get("expected_domain"),
                expected_temporal_mode=expected_temporal_mode,
                expected_min_observations=expected_min_obs,
                expected_primary_indicator=q_dict.get("expected_primary_indicator"),
                expected_source=q_dict.get("expected_source"),
                expected_destination=q_dict.get("expected_destination"),
                expected_time_start=q_dict.get("expected_time_start"),
                expected_time_end=q_dict.get("expected_time_end"),
                expected_properties=q_dict.get("expected_properties", []),
                category=q_dict.get("category", "core_deterministic"),
                metadata=q_dict.get("metadata", {}),
            )
        )

    return golden_queries
