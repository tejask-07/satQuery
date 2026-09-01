import os
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    """
    Load environment variables from a .env file into os.environ
    if they are not already set.
    """
    if not env_path.is_file():
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    # Strip wrapping quotes if any
                    if (val.startswith('"') and val.endswith('"')) or (
                        val.startswith("'") and val.endswith("'")
                    ):
                        val = val[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception as exc:
        print(f"[CONFIG WARNING] Could not load .env from {env_path}: {exc}")


# Find project and backend directories
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

# Load backend/.env first, then root .env if present
load_env_file(BACKEND_DIR / ".env")
load_env_file(PROJECT_DIR / ".env")

# Configured Settings
SENTINEL_STAC_URL = os.getenv(
    "SENTINEL_STAC_URL",
    "https://planetarycomputer.microsoft.com/api/stac/v1",
)
SENTINEL_SAS_SIGN_URL = os.getenv(
    "SENTINEL_SAS_SIGN_URL",
    "https://planetarycomputer.microsoft.com/api/sas/v1/sign",
)
SATQUERY_CACHE_DIR = os.getenv(
    "SATQUERY_CACHE_DIR",
    "backend/data/cache",
)
