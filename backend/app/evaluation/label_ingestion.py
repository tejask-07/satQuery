from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window


# ============================================================
# WORLDCOVER CLASS DEFINITIONS
# ============================================================

# ESA WorldCover class values
WC_TREE_COVER = 10
WC_SHRUBLAND = 20
WC_GRASSLAND = 30
WC_CROPLAND = 40
WC_BUILT_UP = 50
WC_BARE_SPARSE = 60
WC_SNOW_ICE = 70
WC_WATER = 80
WC_WETLAND = 90
WC_MANGROVE = 95
WC_MOSS = 100
WC_NODATA = 0

# SatQuery canonical change classes
SATQUERY_NO_CHANGE = 0
SATQUERY_URBAN_EXPANSION = 1
SATQUERY_URBAN_REDUCTION = 2
SATQUERY_VEGETATION_LOSS = 3
SATQUERY_VEGETATION_GAIN = 4
SATQUERY_WATER_LOSS = 5
SATQUERY_WATER_GAIN = 6
SATQUERY_AMBIGUOUS = 7
SATQUERY_INVALID = 8

# WorldCover semantic groups
WC_VEGETATION = {
    WC_TREE_COVER,
    WC_SHRUBLAND,
    WC_GRASSLAND,
    WC_CROPLAND,
    WC_WETLAND,
    WC_MANGROVE,
    WC_MOSS,
}

WC_NON_VEGETATED = {
    WC_BARE_SPARSE,
    WC_BUILT_UP,
    WC_WATER,
}

# Snow/ice and source NoData are treated as unusable for this benchmark.
WC_INVALID = {
    WC_NODATA,
    WC_SNOW_ICE,
}


# ============================================================
# ESA WORLDCOVER URL BUILDER
# ============================================================

def _worldcover_tile_name(lon_min: float, lat_min: float) -> str:
    """
    Compute the ESA WorldCover 3x3-degree tile name from the AOI
    lower-left coordinate.

    Example:
        lat=48.x, lon=16.x -> N48E015

    WorldCover tiles are identified using their southwest tile corner.
    """
    tile_lat = int(math.floor(lat_min / 3.0) * 3)
    tile_lon = int(math.floor(lon_min / 3.0) * 3)

    lat_prefix = "N" if tile_lat >= 0 else "S"
    lon_prefix = "E" if tile_lon >= 0 else "W"

    lat_str = f"{abs(tile_lat):02d}"
    lon_str = f"{abs(tile_lon):03d}"

    return f"{lat_prefix}{lat_str}{lon_prefix}{lon_str}"


def worldcover_url(
    lat_min: float,
    lon_min: float,
    year: int,
) -> str:
    """
    Return the raw HTTPS URL for the ESA WorldCover COG tile
    covering the given AOI.

    Supported years:
        2020 -> v100
        2021 -> v200
    """
    tile = _worldcover_tile_name(lon_min, lat_min)

    if year == 2020:
        return (
            "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
            f"v100/2020/map/"
            f"ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"
        )

    if year == 2021:
        return (
            "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
            f"v200/2021/map/"
            f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        )

    raise ValueError(
        f"WorldCover is only supported for 2020 and 2021; got year={year}"
    )


# ============================================================
# WORLDCOVER COG STREAMING
# ============================================================

def fetch_worldcover_tile(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    year: int,
    output_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Affine, str]:
    """
    Fetch and optionally cache an AOI subset from the ESA WorldCover COG.

    Returns:
        data_array:
            2D uint8 WorldCover class array.
        transform:
            Affine transform for the returned subset.
        crs_string:
            CRS string, normally EPSG:4326.
    """
    if year not in (2020, 2021):
        raise ValueError(
            f"WorldCover is only supported for 2020 and 2021; got year={year}"
        )

    if output_path is not None:
        output_path = Path(output_path)

        if output_path.exists():
            print(
                f"  [WorldCover] Loading cached real tile from: "
                f"{output_path}"
            )

            with rasterio.open(output_path) as src:
                data = src.read(1)

                return (
                    data,
                    src.transform,
                    str(src.crs),
                )

    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

    url = worldcover_url(
        lat_min=lat_min,
        lon_min=lon_min,
        year=year,
    )

    print(f"  [WorldCover] Fetching {year} tile:")
    print(f"    URL: {url}")
    print(
        "    AOI: "
        f"lon=[{lon_min}, {lon_max}], "
        f"lat=[{lat_min}, {lat_max}]"
    )

    try:
        with rasterio.open(url) as src:
            # Convert geographic AOI bounds into source raster indices.
            row_top, col_left = src.index(
                lon_min,
                lat_max,
            )

            row_bottom, col_right = src.index(
                lon_max,
                lat_min,
            )

            # Clamp to source raster extent.
            row_top = max(0, int(row_top))
            col_left = max(0, int(col_left))

            row_bottom = min(
                src.height,
                int(row_bottom) + 1,
            )

            col_right = min(
                src.width,
                int(col_right) + 1,
            )

            width = col_right - col_left
            height = row_bottom - row_top

            if width <= 0 or height <= 0:
                raise ValueError(
                    "WorldCover AOI produced an empty source window: "
                    f"width={width}, height={height}"
                )

            window = Window(
                col_off=col_left,
                row_off=row_top,
                width=width,
                height=height,
            )

            data = src.read(
                1,
                window=window,
            )

            window_transform = src.window_transform(window)
            crs_str = str(src.crs)

            print(
                f"  [WorldCover] Retrieved shape: {data.shape}"
            )
            print(
                "  [WorldCover] Unique source classes: "
                f"{np.unique(data).tolist()}"
            )

            if output_path is not None:
                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                profile = {
                    "driver": "GTiff",
                    "dtype": rasterio.uint8,
                    "width": data.shape[1],
                    "height": data.shape[0],
                    "count": 1,
                    "crs": src.crs,
                    "transform": window_transform,
                    "compress": "lzw",
                    "nodata": WC_NODATA,
                }

                with rasterio.open(
                    output_path,
                    "w",
                    **profile,
                ) as dst:
                    dst.write(
                        data.astype(np.uint8),
                        1,
                    )

                print(
                    f"  [WorldCover] Saved subset to: {output_path}"
                )

            return (
                data.astype(np.uint8),
                window_transform,
                crs_str,
            )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch ESA WorldCover {year} tile "
            f"for AOI [{lon_min}, {lat_min}, {lon_max}, {lat_max}]: "
            f"{exc}"
        ) from exc


# ============================================================
# BI-TEMPORAL CLASS MAPPING ENGINE
# ============================================================

class WorldCoverMapper:
    """
    Maps WorldCover before/after class pairs into SatQuery's
    canonical change classes.

    IMPORTANT:
    These are derived reference labels, not absolute ground truth.

    Priority:
        1. invalid
        2. no-change
        3. urban transitions
        4. vegetation transitions
        5. water transitions
        6. ambiguous
    """

    LABEL_TYPE = "derived_reference"
    DATASET_NAME = "ESA WorldCover 2020/2021"
    LICENSE = "CC-BY-4.0"

    ALGORITHM_CHANGE_WARNING = (
        "ESA WorldCover 2020 (v100) and 2021 (v200) use "
        "different classification algorithms. Some observed "
        "differences may therefore reflect algorithm/version "
        "changes rather than real land-cover changes. "
        "Labels are derived_reference, not authoritative ground truth."
    )

    @staticmethod
    def map_pixel(
        wc_before: int,
        wc_after: int,
    ) -> int:
        """
        Map one WorldCover before/after pair to a SatQuery class.
        """

        # ----------------------------------------------------
        # Invalid
        # ----------------------------------------------------
        if (
            wc_before in WC_INVALID
            or wc_after in WC_INVALID
        ):
            return SATQUERY_INVALID

        # ----------------------------------------------------
        # No change
        # ----------------------------------------------------
        if wc_before == wc_after:
            return SATQUERY_NO_CHANGE

        # ----------------------------------------------------
        # Urban transitions
        # ----------------------------------------------------
        if (
            wc_after == WC_BUILT_UP
            and wc_before != WC_BUILT_UP
        ):
            return SATQUERY_URBAN_EXPANSION

        if (
            wc_before == WC_BUILT_UP
            and wc_after != WC_BUILT_UP
        ):
            return SATQUERY_URBAN_REDUCTION

        # ----------------------------------------------------
        # Tree loss
        # ----------------------------------------------------
        if (
            wc_before == WC_TREE_COVER
            and wc_after
            in {
                WC_CROPLAND,
                WC_BARE_SPARSE,
                WC_SHRUBLAND,
                WC_GRASSLAND,
            }
        ):
            return SATQUERY_VEGETATION_LOSS

        # ----------------------------------------------------
        # Tree gain
        # ----------------------------------------------------
        if (
            wc_after == WC_TREE_COVER
            and wc_before
            in {
                WC_CROPLAND,
                WC_BARE_SPARSE,
                WC_GRASSLAND,
            }
        ):
            return SATQUERY_VEGETATION_GAIN

        # ----------------------------------------------------
        # General vegetation gain (bare/grass -> grass/shrub)
        # ----------------------------------------------------
        if (
            wc_before
            in {
                WC_BARE_SPARSE,
                WC_GRASSLAND,
            }
            and wc_after
            in {
                WC_GRASSLAND,
                WC_SHRUBLAND,
            }
        ):
            return SATQUERY_VEGETATION_GAIN

        # ----------------------------------------------------
        # Water loss
        # ----------------------------------------------------
        if (
            wc_before == WC_WATER
            and wc_after
            in {
                WC_BARE_SPARSE,
                WC_GRASSLAND,
                WC_CROPLAND,
            }
        ):
            return SATQUERY_WATER_LOSS

        # ----------------------------------------------------
        # Water gain
        # ----------------------------------------------------
        if (
            wc_after == WC_WATER
            and wc_before
            in {
                WC_BARE_SPARSE,
                WC_GRASSLAND,
                WC_CROPLAND,
            }
        ):
            return SATQUERY_WATER_GAIN

        # ----------------------------------------------------
        # Wetlands/mangroves are ambiguous
        # ----------------------------------------------------
        if (
            WC_WETLAND in (wc_before, wc_after)
            or WC_MANGROVE in (wc_before, wc_after)
        ):
            return SATQUERY_AMBIGUOUS

        # ----------------------------------------------------
        # Remaining vegetation-to-vegetation transitions
        # ----------------------------------------------------
        if (
            wc_before in WC_VEGETATION
            and wc_after in WC_VEGETATION
        ):
            return SATQUERY_AMBIGUOUS

        # ----------------------------------------------------
        # Unknown transition
        # ----------------------------------------------------
        return SATQUERY_AMBIGUOUS

    @classmethod
    def apply(
        cls,
        wc_2020: np.ndarray,
        wc_2021: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized WorldCover before/after mapping.

        Returns:
            uint8 array containing classes 0..8.
        """
        if wc_2020.ndim != 2 or wc_2021.ndim != 2:
            raise ValueError(
                "WorldCover arrays must both be 2D."
            )

        if wc_2020.shape != wc_2021.shape:
            raise ValueError(
                "WorldCover array shape mismatch: "
                f"{wc_2020.shape} vs {wc_2021.shape}"
            )

        before = wc_2020.astype(np.uint8, copy=False)
        after = wc_2021.astype(np.uint8, copy=False)

        # Start with NO_CHANGE.
        change_mask = np.full(
            before.shape,
            SATQUERY_NO_CHANGE,
            dtype=np.uint8,
        )

        # ----------------------------------------------------
        # Invalid
        # ----------------------------------------------------
        invalid_mask = (
            np.isin(
                before,
                list(WC_INVALID),
            )
            | np.isin(
                after,
                list(WC_INVALID),
            )
        )

        change_mask[invalid_mask] = SATQUERY_INVALID

        # ----------------------------------------------------
        # Valid changed pixels
        # ----------------------------------------------------
        valid_changed = (
            (~invalid_mask)
            & (before != after)
        )

        # ----------------------------------------------------
        # Urban expansion
        # ----------------------------------------------------
        urban_expansion = (
            valid_changed
            & (after == WC_BUILT_UP)
            & (before != WC_BUILT_UP)
        )

        change_mask[urban_expansion] = SATQUERY_URBAN_EXPANSION

        # ----------------------------------------------------
        # Urban reduction
        # ----------------------------------------------------
        urban_reduction = (
            valid_changed
            & (before == WC_BUILT_UP)
            & (after != WC_BUILT_UP)
        )

        change_mask[urban_reduction] = SATQUERY_URBAN_REDUCTION

        # ----------------------------------------------------
        # Vegetation loss
        # ----------------------------------------------------
        vegetation_loss = (
            valid_changed
            & (before == WC_TREE_COVER)
            & np.isin(
                after,
                [
                    WC_CROPLAND,
                    WC_BARE_SPARSE,
                    WC_SHRUBLAND,
                    WC_GRASSLAND,
                ],
            )
        )

        change_mask[vegetation_loss] = SATQUERY_VEGETATION_LOSS

        # ----------------------------------------------------
        # Vegetation gain
        # ----------------------------------------------------
        vegetation_gain_tree = (
            valid_changed
            & (after == WC_TREE_COVER)
            & np.isin(
                before,
                [
                    WC_CROPLAND,
                    WC_BARE_SPARSE,
                    WC_GRASSLAND,
                ],
            )
        )

        change_mask[vegetation_gain_tree] = SATQUERY_VEGETATION_GAIN

        # ----------------------------------------------------
        # Additional vegetation gain
        # ----------------------------------------------------
        vegetation_gain_general = (
            valid_changed
            & np.isin(
                before,
                [
                    WC_BARE_SPARSE,
                    WC_GRASSLAND,
                ],
            )
            & np.isin(
                after,
                [
                    WC_GRASSLAND,
                    WC_SHRUBLAND,
                ],
            )
            & (change_mask == SATQUERY_NO_CHANGE)
        )

        change_mask[vegetation_gain_general] = SATQUERY_VEGETATION_GAIN

        # ----------------------------------------------------
        # Water loss
        # ----------------------------------------------------
        water_loss = (
            valid_changed
            & (before == WC_WATER)
            & np.isin(
                after,
                [
                    WC_BARE_SPARSE,
                    WC_GRASSLAND,
                    WC_CROPLAND,
                ],
            )
        )

        change_mask[water_loss] = SATQUERY_WATER_LOSS

        # ----------------------------------------------------
        # Water gain
        # ----------------------------------------------------
        water_gain = (
            valid_changed
            & (after == WC_WATER)
            & np.isin(
                before,
                [
                    WC_BARE_SPARSE,
                    WC_GRASSLAND,
                    WC_CROPLAND,
                ],
            )
        )

        change_mask[water_gain] = SATQUERY_WATER_GAIN

        # ----------------------------------------------------
        # Any remaining valid class transition = ambiguous
        # ----------------------------------------------------
        remaining = (
            valid_changed
            & (change_mask == SATQUERY_NO_CHANGE)
        )

        change_mask[remaining] = SATQUERY_AMBIGUOUS

        return change_mask


# ============================================================
# GRID ALIGNMENT ENGINE
# ============================================================

def align_reference_to_grid(
    reference_arr: np.ndarray,
    reference_transform: Affine,
    reference_crs: str,
    target_transform: Affine,
    target_crs: str,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Reproject a categorical reference raster onto the exact SatQuery grid.

    CRITICAL SEMANTIC RULE:
        SATQUERY_NO_CHANGE = 0 is a VALID CLASS.

    Therefore:
        src_nodata MUST NOT be 0.

    SATQUERY_INVALID = 8 is the only SatQuery destination NoData value.
    """

    if reference_arr.ndim != 2:
        raise ValueError(
            f"reference_arr must be 2D, got ndim={reference_arr.ndim}"
        )

    height, width = target_shape

    if height <= 0 or width <= 0:
        raise ValueError(
            f"Invalid target shape: {target_shape}"
        )

    # Verify source labels are valid canonical values.
    unique_source = np.unique(reference_arr)

    invalid_source_values = [
        int(value)
        for value in unique_source
        if int(value) < SATQUERY_NO_CHANGE
        or int(value) > SATQUERY_INVALID
    ]

    if invalid_source_values:
        raise ValueError(
            "Reference array contains invalid SatQuery class values: "
            f"{invalid_source_values}"
        )

    aligned = np.full(
        (height, width),
        SATQUERY_INVALID,
        dtype=np.uint8,
    )

    source_crs = (
        reference_crs
        if isinstance(reference_crs, CRS)
        else CRS.from_string(reference_crs)
    )

    destination_crs = (
        target_crs
        if isinstance(target_crs, CRS)
        else CRS.from_string(target_crs)
    )

    reproject(
        source=reference_arr.astype(np.uint8, copy=False),
        destination=aligned,
        src_transform=reference_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=destination_crs,
        resampling=Resampling.nearest,

        # IMPORTANT:
        # 0 = valid NO_CHANGE, therefore it must NOT be source NoData.
        src_nodata=SATQUERY_INVALID,
        dst_nodata=SATQUERY_INVALID,
    )

    if aligned.shape != (height, width):
        raise ValueError(
            "Alignment produced unexpected shape: "
            f"{aligned.shape}; expected {(height, width)}"
        )

    return aligned


# ============================================================
# REFERENCE LABEL VALIDATION
# ============================================================

@dataclass
class AlignmentValidationResult:
    is_valid: bool
    shape: Tuple[int, int]
    unique_classes: List[int]
    class_distribution: Dict[int, int]
    evaluated_pixel_count: int
    invalid_pixel_count: int
    valid_pixel_fraction: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_aligned_reference(
    aligned_arr: np.ndarray,
    target_shape: Tuple[int, int],
    min_valid_fraction: float = 0.10,
) -> AlignmentValidationResult:
    """
    Validate an aligned SatQuery reference change mask.

    Class 8 is invalid and excluded from evaluation.

    Class 0 is VALID no-change and is therefore included in the
    evaluated-pixel count.
    """

    issues: List[str] = []
    warnings: List[str] = []

    aligned_arr = np.asarray(aligned_arr)

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------
    if aligned_arr.shape != target_shape:
        issues.append(
            f"Shape mismatch: got {aligned_arr.shape}, "
            f"expected {target_shape}"
        )

        return AlignmentValidationResult(
            is_valid=False,
            shape=aligned_arr.shape,
            unique_classes=[],
            class_distribution={},
            evaluated_pixel_count=0,
            invalid_pixel_count=aligned_arr.size,
            valid_pixel_fraction=0.0,
            issues=issues,
            warnings=warnings,
        )

    # --------------------------------------------------------
    # Class range
    # --------------------------------------------------------
    unique_vals = np.unique(aligned_arr).tolist()

    invalid_labels = [
        int(value)
        for value in unique_vals
        if int(value) < SATQUERY_NO_CHANGE
        or int(value) > SATQUERY_INVALID
    ]

    if invalid_labels:
        issues.append(
            "Reference contains out-of-range class values: "
            f"{invalid_labels}"
        )

    # --------------------------------------------------------
    # Valid/invalid counts
    # --------------------------------------------------------
    invalid_count = int(
        np.sum(aligned_arr == SATQUERY_INVALID)
    )

    evaluated_count = int(
        aligned_arr.size - invalid_count
    )

    valid_fraction = (
        evaluated_count / aligned_arr.size
        if aligned_arr.size > 0
        else 0.0
    )

    if evaluated_count == 0:
        issues.append(
            "Zero valid pixels in aligned reference."
        )

    elif valid_fraction < min_valid_fraction:
        warnings.append(
            "Low valid pixel fraction: "
            f"{valid_fraction:.1%} "
            f"(threshold: {min_valid_fraction:.1%})"
        )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------
    class_distribution = {
        class_id: int(
            np.sum(aligned_arr == class_id)
        )
        for class_id in range(9)
    }

    # 0 is valid no-change, so only warn if ALL pixels,
    # including invalid pixels, are represented by class 0.
    if (
        class_distribution.get(
            SATQUERY_NO_CHANGE,
            0,
        )
        == aligned_arr.size
    ):
        warnings.append(
            "All pixels are class 0 (no_change). "
            "Possible blank or unchanged reference."
        )

    return AlignmentValidationResult(
        is_valid=(len(issues) == 0),
        shape=aligned_arr.shape,
        unique_classes=[
            int(value)
            for value in unique_vals
        ],
        class_distribution=class_distribution,
        evaluated_pixel_count=evaluated_count,
        invalid_pixel_count=invalid_count,
        valid_pixel_fraction=round(
            valid_fraction,
            4,
        ),
        issues=issues,
        warnings=warnings,
    )


# ============================================================
# MATERIALIZATION RESULT
# ============================================================

@dataclass
class MaterializationResult:
    example_id: str
    status: str
    ground_truth_path: Optional[str]
    label_source: str
    label_type: str
    label_version: str
    class_mapping_schema: str
    reference_crs: str
    reference_resolution_m: float
    processing_method: str
    validation_result: Optional[AlignmentValidationResult]
    validation_notes: str
    error_message: Optional[str] = None
    provenance: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# END-TO-END WORLDCOVER MATERIALIZATION
# ============================================================

def materialize_worldcover_example(
    example_id: str,
    aoi_lon_min: float,
    aoi_lat_min: float,
    aoi_lon_max: float,
    aoi_lat_max: float,
    target_scene_path: str,
    output_dir: Path,
    reference_dir: Optional[Path] = None,
    year_before: int = 2020,
    year_after: int = 2021,
) -> MaterializationResult:
    """
    End-to-end WorldCover reference-label materialization.

    Steps:
        1. Fetch WorldCover before/after subsets.
        2. Map WorldCover transitions into SatQuery classes.
        3. Load target SatQuery grid.
        4. Align reference labels using nearest-neighbor.
        5. Validate aligned labels.
        6. Save aligned GeoTIFF with NoData=8.
    """

    start_time = time.time()

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aligned_path = (
        output_dir
        / f"{example_id}_aligned.tif"
    )

    print(
        f"\n[Materialization] Starting: {example_id}"
    )

    print(
        "  AOI: "
        f"lon=[{aoi_lon_min:.6f}, {aoi_lon_max:.6f}], "
        f"lat=[{aoi_lat_min:.6f}, {aoi_lat_max:.6f}]"
    )

    try:
        # ----------------------------------------------------
        # Step 1: Reference directories
        # ----------------------------------------------------
        raw_before_path: Optional[Path] = None
        raw_after_path: Optional[Path] = None

        if reference_dir is not None:
            reference_dir = Path(reference_dir)

            reference_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            raw_before_path = (
                reference_dir
                / f"wc_{year_before}_{example_id}.tif"
            )

            raw_after_path = (
                reference_dir
                / f"wc_{year_after}_{example_id}.tif"
            )

        # ----------------------------------------------------
        # Step 2: Fetch real WorldCover
        # ----------------------------------------------------
        (
            wc_before,
            wc_before_transform,
            wc_before_crs,
        ) = fetch_worldcover_tile(
            lon_min=aoi_lon_min,
            lat_min=aoi_lat_min,
            lon_max=aoi_lon_max,
            lat_max=aoi_lat_max,
            year=year_before,
            output_path=raw_before_path,
        )

        (
            wc_after,
            wc_after_transform,
            wc_after_crs,
        ) = fetch_worldcover_tile(
            lon_min=aoi_lon_min,
            lat_min=aoi_lat_min,
            lon_max=aoi_lon_max,
            lat_max=aoi_lat_max,
            year=year_after,
            output_path=raw_after_path,
        )

        # WorldCover yearly subsets must represent the same source grid.
        if wc_before.shape != wc_after.shape:
            raise ValueError(
                "WorldCover before/after shape mismatch: "
                f"{wc_before.shape} vs {wc_after.shape}"
            )

        if str(wc_before_crs) != str(wc_after_crs):
            raise ValueError(
                "WorldCover before/after CRS mismatch: "
                f"{wc_before_crs} vs {wc_after_crs}"
            )

        # ----------------------------------------------------
        # Step 3: Verify source transforms
        # ----------------------------------------------------
        transform_difference = np.max(
            np.abs(
                np.asarray(wc_before_transform)
                - np.asarray(wc_after_transform)
            )
        )

        if transform_difference > 1e-9:
            raise ValueError(
                "WorldCover before/after transforms are not identical: "
                f"max difference={transform_difference}"
            )

        # ----------------------------------------------------
        # Step 4: Map transitions
        # ----------------------------------------------------
        print(
            "  [Mapping] Applying WorldCover bi-temporal mapping..."
        )

        change_mask = WorldCoverMapper.apply(
            wc_before,
            wc_after,
        )

        unique_change_classes = [
            int(value)
            for value in np.unique(change_mask)
        ]

        print(
            "  [Mapping] Change-mask shape: "
            f"{change_mask.shape}"
        )

        print(
            "  [Mapping] SatQuery classes: "
            f"{unique_change_classes}"
        )

        # ----------------------------------------------------
        # Step 5: Load target SatQuery grid
        # ----------------------------------------------------
        target_scene_path = str(
            Path(target_scene_path)
        )

        if not Path(target_scene_path).exists():
            raise FileNotFoundError(
                f"Target SatQuery scene raster does not exist: "
                f"{target_scene_path}"
            )

        with rasterio.open(
            target_scene_path
        ) as src:

            target_transform = src.transform
            target_crs = str(src.crs)
            target_shape = src.shape
            target_resolution = float(
                abs(src.transform.a)
            )

        print(
            "  [Alignment] Target grid: "
            f"shape={target_shape}, "
            f"CRS={target_crs}, "
            f"resolution~{target_resolution:.4f}"
        )

        # ----------------------------------------------------
        # Step 6: Align reference to target grid
        # ----------------------------------------------------
        aligned = align_reference_to_grid(
            reference_arr=change_mask,
            reference_transform=wc_before_transform,
            reference_crs=str(wc_before_crs),
            target_transform=target_transform,
            target_crs=target_crs,
            target_shape=target_shape,
        )

        print(
            "  [Alignment] Aligned shape: "
            f"{aligned.shape}"
        )

        print(
            "  [Alignment] Aligned classes: "
            f"{np.unique(aligned).tolist()}"
        )

        # ----------------------------------------------------
        # Step 7: Validate
        # ----------------------------------------------------
        validation_result = validate_aligned_reference(
            aligned_arr=aligned,
            target_shape=target_shape,
        )

        print(
            "  [Validation] "
            f"valid={validation_result.is_valid}"
        )

        print(
            "  [Validation] "
            f"evaluated="
            f"{validation_result.evaluated_pixel_count}"
            f"/{aligned.size}"
            f" "
            f"({validation_result.valid_pixel_fraction:.1%})"
        )

        print(
            "  [Validation] Class distribution: "
            f"{validation_result.class_distribution}"
        )

        for issue in validation_result.issues:
            print(
                f"  [Validation] ERROR: {issue}"
            )

        for warning in validation_result.warnings:
            print(
                f"  [Validation] WARNING: {warning}"
            )

        if not validation_result.is_valid:
            return MaterializationResult(
                example_id=example_id,
                status="failed",
                ground_truth_path=None,
                label_source=(
                    "ESA WorldCover 2020/2021"
                ),
                label_type="derived_reference",
                label_version=(
                    "v100_2020_v200_2021"
                ),
                class_mapping_schema=(
                    "ESA_WorldCover_to_SatQuery_v1"
                ),
                reference_crs=str(
                    wc_before_crs
                ),
                reference_resolution_m=10.0,
                processing_method=(
                    "worldcover_bitemporal_differencing"
                ),
                validation_result=validation_result,
                validation_notes="; ".join(
                    validation_result.issues
                ),
                error_message=(
                    "Alignment validation failed."
                ),
            )

        # ----------------------------------------------------
        # Step 8: Save aligned reference
        # ----------------------------------------------------
        target_crs_obj = CRS.from_string(
            target_crs
        )

        output_profile = {
            "driver": "GTiff",
            "dtype": rasterio.uint8,
            "width": target_shape[1],
            "height": target_shape[0],
            "count": 1,
            "crs": target_crs_obj,
            "transform": target_transform,
            "compress": "lzw",

            # IMPORTANT:
            # 0 = valid no_change
            # 8 = invalid/nodata
            "nodata": SATQUERY_INVALID,
        }

        with rasterio.open(
            aligned_path,
            "w",
            **output_profile,
        ) as dst:

            dst.write(
                aligned.astype(np.uint8),
                1,
            )

            dst.update_tags(
                example_id=example_id,
                label_source=(
                    "ESA WorldCover 2020/2021"
                ),
                label_type="derived_reference",
                label_version=(
                    "v100_2020_v200_2021"
                ),
                class_mapping_schema=(
                    "ESA_WorldCover_to_SatQuery_v1"
                ),
                processing_method=(
                    "worldcover_bitemporal_differencing"
                ),
                algorithm_change_warning=(
                    WorldCoverMapper.ALGORITHM_CHANGE_WARNING
                ),
                reference_crs=str(
                    wc_before_crs
                ),
                aligned_crs=target_crs,
                reference_resolution_m="10.0",
                aligned_resolution_m=(
                    str(target_resolution)
                ),
                generated_by=(
                    "SatQuery Phase 9C "
                    "label_ingestion.py"
                ),
                generation_date=(
                    time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(),
                    )
                ),
            )

        elapsed = round(
            time.time() - start_time,
            2,
        )

        print(
            f"  [Done] Materialized in {elapsed}s:"
            f" {aligned_path}"
        )

        validation_note_parts: List[str] = []

        if validation_result.warnings:
            validation_note_parts.extend(
                validation_result.warnings
            )

        validation_note_parts.append(
            WorldCoverMapper.ALGORITHM_CHANGE_WARNING
        )

        return MaterializationResult(
            example_id=example_id,
            status="materialized",
            ground_truth_path=str(
                aligned_path
            ),
            label_source=(
                "ESA WorldCover 2020/2021"
            ),
            label_type="derived_reference",
            label_version=(
                "v100_2020_v200_2021"
            ),
            class_mapping_schema=(
                "ESA_WorldCover_to_SatQuery_v1"
            ),
            reference_crs=str(
                wc_before_crs
            ),
            reference_resolution_m=10.0,
            processing_method=(
                "worldcover_bitemporal_differencing"
            ),
            validation_result=validation_result,
            validation_notes=" | ".join(
                validation_note_parts
            ),
            provenance={
                "worldcover_2020_url": worldcover_url(
                    aoi_lat_min,
                    aoi_lon_min,
                    year_before,
                ),
                "worldcover_2021_url": worldcover_url(
                    aoi_lat_min,
                    aoi_lon_min,
                    year_after,
                ),
                "wc_before_shape": list(
                    wc_before.shape
                ),
                "wc_after_shape": list(
                    wc_after.shape
                ),
                "wc_before_crs": str(
                    wc_before_crs
                ),
                "wc_after_crs": str(
                    wc_after_crs
                ),
                "change_mask_unique_classes": (
                    unique_change_classes
                ),
                "target_scene_reference": (
                    target_scene_path
                ),
                "target_crs": target_crs,
                "target_shape": list(
                    target_shape
                ),
                "target_resolution_m": (
                    target_resolution
                ),
                "aligned_nodata": SATQUERY_INVALID,
                "aoi_bounds": [
                    aoi_lon_min,
                    aoi_lat_min,
                    aoi_lon_max,
                    aoi_lat_max,
                ],
                "processing_elapsed_s": elapsed,
                "license": "CC-BY-4.0",
                "citation": (
                    "Zanaga, D., et al. (2022). "
                    "ESA WorldCover 10m 2020 v100. "
                    "doi:10.5281/zenodo.5571936"
                ),
                "reference_label_qualification": (
                    "derived_reference"
                ),
                "algorithm_change_warning": (
                    WorldCoverMapper.ALGORITHM_CHANGE_WARNING
                ),
            },
        )

    except Exception as exc:
        print(
            f"  [ERROR] Materialization failed: {exc}"
        )

        return MaterializationResult(
            example_id=example_id,
            status="failed",
            ground_truth_path=None,
            label_source=(
                "ESA WorldCover 2020/2021"
            ),
            label_type="derived_reference",
            label_version=(
                "v100_2020_v200_2021"
            ),
            class_mapping_schema=(
                "ESA_WorldCover_to_SatQuery_v1"
            ),
            reference_crs="EPSG:4326",
            reference_resolution_m=10.0,
            processing_method=(
                "worldcover_bitemporal_differencing"
            ),
            validation_result=None,
            validation_notes="",
            error_message=str(exc),
        )


# ============================================================
# OSCD REJECTION DOCUMENTATION
# ============================================================

OSCD_REJECTION_RECORD = {
    "dataset": (
        "OSCD (Onera Satellite Change Detection)"
    ),
    "status": "REJECTED_NO_SCENE_MATCH",
    "rejection_reason": (
        "The current benchmark AOIs do not have "
        "a matching OSCD scene. OSCD contains a "
        "different set of cities and therefore "
        "cannot be used as a reference label for "
        "the current Vienna/Delhi/Queensland "
        "benchmark examples without changing the "
        "benchmark scene set."
    ),
    "assessed_by": (
        "SatQuery Phase 9C label_ingestion.py"
    ),
    "assessment_date": "2026-09-04",
    "source_url": (
        "https://ieee-dataport.org/open-access/"
        "oscd-onera-satellite-change-detection"
    ),
    "license": "CC-BY-NC-SA-4.0",
}