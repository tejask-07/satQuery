from pathlib import Path

import cv2
import numpy as np


VISUALIZATION_DIR = (
    Path(__file__).resolve().parent
    / "visualizations"
)


def _build_change_colors(
    change_map: np.ndarray,
) -> np.ndarray:
    """
    Convert a numeric change map into a readable color map.

    Interpretation:

        negative change -> red
        zero / very small change -> gray
        positive change -> green
        invalid / NaN -> black

    The magnitude of the change controls brightness.
    """

    change_map = np.asarray(
        change_map,
        dtype=np.float32,
    )

    height, width = change_map.shape

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    valid = np.isfinite(change_map)

    if not np.any(valid):
        return image

    values = change_map[valid]

    max_abs = float(
        np.max(np.abs(values))
    )

    if max_abs <= 0:
        image[valid] = (
            128,
            128,
            128,
        )
        return image

    magnitude = (
        np.abs(change_map)
        / max_abs
    )

    magnitude = np.clip(
        magnitude,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Small changes are rendered gray.
    # Larger changes become increasingly saturated.
    # --------------------------------------------------------

    small_change_threshold = 0.05

    negative = (
        valid
        & (change_map < -small_change_threshold)
    )

    positive = (
        valid
        & (change_map > small_change_threshold)
    )

    neutral = (
        valid
        & ~negative
        & ~positive
    )

    # --------------------------------------------------------
    # Neutral pixels
    # --------------------------------------------------------

    neutral_value = 110

    image[neutral] = (
        neutral_value,
        neutral_value,
        neutral_value,
    )

    # --------------------------------------------------------
    # Negative change
    #
    # OpenCV uses BGR:
    # red = (0, 0, 255)
    # --------------------------------------------------------

    negative_intensity = (
        80
        + 175 * magnitude[negative]
    ).astype(np.uint8)

    image[negative, 0] = 40
    image[negative, 1] = 40
    image[negative, 2] = negative_intensity

    # --------------------------------------------------------
    # Positive change
    #
    # green = (0, 255, 0)
    # --------------------------------------------------------

    positive_intensity = (
        80
        + 175 * magnitude[positive]
    ).astype(np.uint8)

    image[positive, 0] = 40
    image[positive, 1] = positive_intensity
    image[positive, 2] = 40

    # --------------------------------------------------------
    # Invalid pixels
    # --------------------------------------------------------

    image[~valid] = (
        0,
        0,
        0,
    )

    return image


def _resize_map(
    image: np.ndarray,
    target_width: int = 720,
    target_height: int = 500,
) -> np.ndarray:
    """
    Resize a raster visualization while preserving
    individual pixel boundaries.

    Nearest-neighbour interpolation is important for
    satellite/change rasters because we do not want
    artificial smoothing between pixels.
    """

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(
            "Visualization image has invalid dimensions."
        )

    scale = min(
        target_width / width,
        target_height / height,
    )

    new_width = max(
        1,
        int(round(width * scale)),
    )

    new_height = max(
        1,
        int(round(height * scale)),
    )

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )


def _add_title(
    canvas: np.ndarray,
    filename: str,
) -> None:
    """
    Add a readable title derived from the output filename.
    """

    title = Path(filename).stem

    title = title.replace(
        "_",
        " ",
    )

    title = title.replace(
        "change",
        "Change",
    )

    title = title.title()

    cv2.putText(
        canvas,
        title,
        (40, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )


def _add_legend(
    canvas: np.ndarray,
    y: int,
) -> None:
    """
    Add a simple legend explaining the visualization.
    """

    items = [
        (
            "Decrease",
            (40, 40, 220),
        ),
        (
            "No significant change",
            (110, 110, 110),
        ),
        (
            "Increase",
            (40, 220, 40),
        ),
        (
            "No data",
            (0, 0, 0),
        ),
    ]

    x = 40

    for label, color in items:

        cv2.rectangle(
            canvas,
            (x, y - 15),
            (x + 24, y + 9),
            color,
            thickness=-1,
        )

        cv2.rectangle(
            canvas,
            (x, y - 15),
            (x + 24, y + 9),
            (170, 170, 170),
            thickness=1,
        )

        cv2.putText(
            canvas,
            label,
            (x + 34, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        x += (
            175
            if label != "No significant change"
            else 230
        )


def save_change_map(
    change_map: np.ndarray,
    filename: str,
) -> str:
    """
    Save a change-map array as a presentation-ready PNG.

    The original raster dimensions are preserved conceptually,
    but the visualization is enlarged using nearest-neighbour
    interpolation so small test rasters remain readable.

    Interpretation:

        red   -> negative change
        gray  -> no significant change
        green -> positive change
        black -> invalid / no data

    Returns
    -------
    str
        Absolute path to the generated PNG.
    """

    VISUALIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    change_map = np.asarray(
        change_map,
        dtype=np.float32,
    )

    if change_map.ndim != 2:
        raise ValueError(
            "change_map must be a 2D array."
        )

    height, width = change_map.shape

    if height == 0 or width == 0:
        raise ValueError(
            "change_map cannot be empty."
        )

    # --------------------------------------------------------
    # Convert numeric values into semantic colors.
    # --------------------------------------------------------

    color_map = _build_change_colors(
        change_map
    )

    # --------------------------------------------------------
    # Enlarge the raster.
    # --------------------------------------------------------

    enlarged_map = _resize_map(
        color_map,
        target_width=720,
        target_height=500,
    )

    map_height, map_width = (
        enlarged_map.shape[:2]
    )

    # --------------------------------------------------------
    # Create presentation canvas.
    #
    # Header: 70 px
    # Map:    up to 500 px
    # Legend: 80 px
    # --------------------------------------------------------

    canvas_width = max(
        800,
        map_width + 80,
    )

    canvas_height = (
        70
        + map_height
        + 90
    )

    canvas = np.full(
        (
            canvas_height,
            canvas_width,
            3,
        ),
        25,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    _add_title(
        canvas,
        filename,
    )

    # --------------------------------------------------------
    # Put raster in the center.
    # --------------------------------------------------------

    map_x = (
        canvas_width
        - map_width
    ) // 2

    map_y = 65

    canvas[
        map_y : map_y + map_height,
        map_x : map_x + map_width,
    ] = enlarged_map

    # --------------------------------------------------------
    # Border around raster.
    # --------------------------------------------------------

    cv2.rectangle(
        canvas,
        (
            map_x,
            map_y,
        ),
        (
            map_x + map_width - 1,
            map_y + map_height - 1,
        ),
        (150, 150, 150),
        1,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_y = (
        map_y
        + map_height
        + 35
    )

    _add_legend(
        canvas,
        legend_y,
    )

    # --------------------------------------------------------
    # Small metadata line.
    # --------------------------------------------------------

    valid = np.isfinite(
        change_map
    )

    valid_count = int(
        np.sum(valid)
    )

    total_count = int(
        change_map.size
    )

    if valid_count > 0:

        min_value = float(
            np.min(
                change_map[valid]
            )
        )

        max_value = float(
            np.max(
                change_map[valid]
            )
        )

        metadata = (
            f"Raster: {width} x {height} pixels  |  "
            f"Valid: {valid_count}/{total_count}  |  "
            f"Range: {min_value:.4f} to {max_value:.4f}"
        )

    else:

        metadata = (
            f"Raster: {width} x {height} pixels  |  "
            "No valid data"
        )

    cv2.putText(
        canvas,
        metadata,
        (40, canvas_height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        VISUALIZATION_DIR / filename
    )

    success = cv2.imwrite(
        str(output_path),
        canvas,
    )

    if not success:
        raise IOError(
            f"Failed to save visualization: "
            f"{output_path}"
        )

    return str(output_path)