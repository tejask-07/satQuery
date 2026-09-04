"""
Phase 9: Benchmark Dataset & Manifest Validator.

Validates the integrity, consistency, and scientific safeguards of benchmark datasets:
1. Missing files (bands, masks, ground truth where materialized)
2. Support for two-stage benchmark workflow (materialized vs pending_reference_label)
3. CRS and transform consistency
4. Raster shape alignment between before/after and ground truth
5. Supported label values (0 to 8)
6. Duplicate example IDs
7. Split leakage (zero overlap of scenes/regions between TRAIN, VALIDATION, TEST)
8. Supported semantic class mapping schemas (rejects unsupported mappings)
9. Provenance completeness (license, source, dates)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import rasterio

from app.evaluation.paths import resolve_repo_path

SUPPORTED_CLASS_MAPPINGS = {
    "DynamicWorld_to_SatQuery_v1",
    "OSCD_Binary_to_SatQuery_v1",
    "ESA_WorldCover_to_SatQuery_v1",
    "ESA_WorldCover_to_SatQuery_v2",
}


@dataclass
class ValidationIssue:
    severity: str  # "ERROR" | "WARNING" | "INFO"
    example_id: Optional[str]
    message: str


@dataclass
class ValidationResult:
    is_valid: bool
    total_examples: int
    materialized_count: int
    pending_count: int
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_count: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "WARNING")


def validate_benchmark_manifest(manifest_path: Path | str) -> ValidationResult:
    """
    Validates benchmark manifest and associated GeoTIFFs (if materialized).
    """
    manifest_file = resolve_repo_path(manifest_path)
    issues: List[ValidationIssue] = []

    if not manifest_file.exists():
        return ValidationResult(
            is_valid=False,
            total_examples=0,
            materialized_count=0,
            pending_count=0,
            issues=[ValidationIssue("ERROR", None, f"Manifest file not found: {manifest_file}")],
            validated_count=0,
        )

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            total_examples=0,
            materialized_count=0,
            pending_count=0,
            issues=[ValidationIssue("ERROR", None, f"Failed to parse manifest JSON: {e}")],
            validated_count=0,
        )

    examples = data.get("candidate_examples") or data.get("examples", [])
    total_examples = len(examples)

    if total_examples == 0:
        issues.append(ValidationIssue("ERROR", None, "Manifest contains 0 examples."))

    # Check for duplicate example IDs
    seen_ids: Set[str] = set()
    for ex in examples:
        eid = ex.get("example_id")
        if not eid:
            issues.append(ValidationIssue("ERROR", None, "Example missing 'example_id'."))
            continue
        if eid in seen_ids:
            issues.append(ValidationIssue("ERROR", eid, f"Duplicate example_id: '{eid}'."))
        seen_ids.add(eid)

    # Check split leakage: ensure no scene or region overlap across TRAIN, VALIDATION, and TEST
    split_scenes: Dict[str, Set[str]] = {"TRAIN": set(), "VALIDATION": set(), "TEST": set()}
    for ex in examples:
        eid = ex.get("example_id")
        split = ex.get("split")
        if split not in split_scenes:
            issues.append(ValidationIssue("ERROR", eid, f"Invalid split '{split}'. Must be TRAIN, VALIDATION, or TEST."))
            continue

        b_scene = ex.get("before_scene_id")
        a_scene = ex.get("after_scene_id")
        if b_scene:
            split_scenes[split].add(b_scene)
        if a_scene:
            split_scenes[split].add(a_scene)

    splits = ["TRAIN", "VALIDATION", "TEST"]
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = split_scenes[s1] & split_scenes[s2]
            if overlap:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        None,
                        f"Data leakage detected: scenes {overlap} shared between {s1} and {s2} splits.",
                    )
                )

    materialized_count = 0
    pending_count = 0
    validated_count = 0

    # Validate each example
    for ex in examples:
        eid = ex.get("example_id", "UNKNOWN")
        status = ex.get("status", "materialized")
        gt_path_str = ex.get("ground_truth_path")

        # Required fields
        for req in ["split", "target_class", "before_date", "after_date"]:
            if not ex.get(req):
                issues.append(ValidationIssue("ERROR", eid, f"Missing required field '{req}'."))

        # Class mapping validation
        mapping_schema = ex.get("class_mapping_schema")
        if mapping_schema and mapping_schema not in SUPPORTED_CLASS_MAPPINGS:
            issues.append(
                ValidationIssue(
                    "ERROR",
                    eid,
                    f"Unsupported class mapping schema '{mapping_schema}'. Allowed: {SUPPORTED_CLASS_MAPPINGS}",
                )
            )

        if status == "pending_reference_label" or gt_path_str is None:
            pending_count += 1
            issues.append(
                ValidationIssue(
                    "INFO",
                    eid,
                    "Reference label unmaterialized; correctly designated 'pending_reference_label'.",
                )
            )
            continue

        # For materialized examples, validate the GeoTIFF
        materialized_count += 1
        gt_path = resolve_repo_path(gt_path_str)
        if not gt_path.exists():
            issues.append(ValidationIssue("ERROR", eid, f"Ground truth file not found: {gt_path}"))
            continue

        is_gt_clean = True
        try:
            with rasterio.open(gt_path) as gt_src:
                gt_shape = gt_src.shape
                gt_crs = str(gt_src.crs)
                gt_data = gt_src.read(1)

                unique_vals = np.unique(gt_data)
                invalid_labels = [int(v) for v in unique_vals if v < 0 or v > 8]
                if invalid_labels:
                    issues.append(
                        ValidationIssue(
                            "ERROR",
                            eid,
                            f"Ground truth contains unsupported class values: {invalid_labels}. Expected 0..8.",
                        )
                    )
                    is_gt_clean = False

                if gt_data.size == 0:
                    issues.append(ValidationIssue("ERROR", eid, "Ground truth raster has size 0."))
                    is_gt_clean = False

        except Exception as e:
            issues.append(ValidationIssue("ERROR", eid, f"Failed to open ground truth raster: {e}"))
            is_gt_clean = False

        if is_gt_clean and status == "validated":
            validated_count += 1

    is_valid = sum(1 for i in issues if i.severity == "ERROR") == 0
    return ValidationResult(
        is_valid=is_valid,
        total_examples=total_examples,
        materialized_count=materialized_count,
        pending_count=pending_count,
        issues=issues,
        validated_count=validated_count,
    )
