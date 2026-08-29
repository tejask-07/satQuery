from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def search_imagery(**kwargs):
    """
    Return mock Sentinel-2 imagery records for the MVP.
    """

    return {
        "status": "success",
        "source": "MOCK_SENTINEL_2",
        "images": [
            {
                "id": "mock_image_2021",
                "date": "2021-06-15",
                "cloud_cover": 8.2,
                "bands": {
                    "red": str(
                        SAMPLES_DIR / "before_red.tif"
                    ),
                    "green": str(
                        SAMPLES_DIR / "before_green.tif"
                    ),
                    "nir": str(
                        SAMPLES_DIR / "before_nir.tif"
                    ),
                    "swir": str(
                        SAMPLES_DIR / "before_swir.tif"
                    ),
                },
            },
            {
                "id": "mock_image_2025",
                "date": "2025-06-18",
                "cloud_cover": 6.7,
                "bands": {
                    "red": str(
                        SAMPLES_DIR / "after_red.tif"
                    ),
                    "green": str(
                        SAMPLES_DIR / "after_green.tif"
                    ),
                    "nir": str(
                        SAMPLES_DIR / "after_nir.tif"
                    ),
                    "swir": str(
                        SAMPLES_DIR / "after_swir.tif"
                    ),
                },
            },
        ],
    }