import io
import sys
import tarfile
from pathlib import Path
from typing import Dict

import numpy as np
import requests
import zstandard as zstd
from rasterio.io import MemoryFile


# ============================================================
# REMOTE FALLBACK
# ============================================================

ZENODO_S1_URL = (
    "https://zenodo.org/api/records/10891137"
    "/files/BigEarthNet-S1.tar.zst/content"
)


EXPECTED_S1_BANDS = {
    "VV",
    "VH",
}


# ============================================================
# LOCAL S1 CACHE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOCAL_S1_CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "s1_cache"
)


# ============================================================
# IN-MEMORY S1 CACHE
# ============================================================

S1_PATCH_CACHE: Dict[
    str,
    Dict[str, np.ndarray]
] = {}


# ============================================================
# HTTP RANGE READER
# ============================================================

class HTTPRangeReader(io.RawIOBase):
    """
    Sequential remote reader using HTTP Range requests.

    The complete Zenodo archive is never stored locally.
    """

    def __init__(
        self,
        url: str,
        chunk_size: int = 8 * 1024 * 1024,
    ):
        self.url = url
        self.chunk_size = chunk_size

        self.position = 0
        self.buffer = b""
        self.buffer_position = 0
        self.total_size = None

    def readable(self):
        return True

    def seekable(self):
        return False

    def tell(self):
        return self.position

    def _fetch_chunk(self):

        start = self.position
        end = (
            start
            + self.chunk_size
            - 1
        )

        response = requests.get(
            self.url,
            headers={
                "Range": (
                    f"bytes={start}-{end}"
                )
            },
            timeout=120,
        )

        response.raise_for_status()

        if response.status_code != 206:
            raise RuntimeError(
                f"Expected HTTP 206, "
                f"got {response.status_code}"
            )

        content_range = (
            response.headers.get(
                "Content-Range"
            )
        )

        print(
            f"HTTP Range: {content_range}"
        )

        self.buffer = response.content
        self.buffer_position = 0

        if (
            content_range
            and "/"
            in content_range
        ):
            try:
                self.total_size = int(
                    content_range.split("/")[-1]
                )
            except ValueError:
                pass

        if not self.buffer:
            raise EOFError(
                "Zenodo returned empty data."
            )

    def read(self, size=-1):

        if size == 0:
            return b""

        output = bytearray()

        while (
            size < 0
            or len(output) < size
        ):

            if (
                self.buffer_position
                >= len(self.buffer)
            ):
                self._fetch_chunk()

            available = (
                len(self.buffer)
                - self.buffer_position
            )

            if size < 0:

                take = available

            else:

                remaining = (
                    size
                    - len(output)
                )

                take = min(
                    available,
                    remaining,
                )

            start = (
                self.buffer_position
            )

            end = start + take

            output.extend(
                self.buffer[start:end]
            )

            self.buffer_position = end
            self.position += take

            if (
                size >= 0
                and len(output) >= size
            ):
                break

        return bytes(output)

    def readinto(self, b):

        data = self.read(len(b))

        n = len(data)

        b[:n] = data

        return n


# ============================================================
# HELPERS
# ============================================================

def make_target_root(
    s1_name: str,
) -> str:

    return (
        "BigEarthNet-S1/"
        f"{s1_name}/"
    )


def band_name_from_path(
    path: str,
) -> str:

    filename = path.rsplit(
        "/",
        1,
    )[-1]

    filename_upper = (
        filename.upper()
    )

    if "_VV" in filename_upper:
        return "VV"

    if "_VH" in filename_upper:
        return "VH"

    return (
        filename
        .rsplit("_", 1)[-1]
        .replace(".tif", "")
    )


def get_local_patch_dir(
    s1_name: str,
) -> Path:

    return (
        LOCAL_S1_CACHE_ROOT
        / s1_name
    )


# ============================================================
# LOAD LOCAL S1 PATCH
# ============================================================

def load_local_s1_bands(
    s1_name: str,
) -> Dict[str, np.ndarray]:

    patch_dir = get_local_patch_dir(
        s1_name
    )

    if not patch_dir.exists():
        raise FileNotFoundError(
            f"Local S1 patch directory "
            f"does not exist:\n{patch_dir}"
        )

    bands: Dict[
        str,
        np.ndarray
    ] = {}

    for path in patch_dir.glob(
        "*.tif"
    ):

        band = band_name_from_path(
            path.name
        )

        if band not in EXPECTED_S1_BANDS:
            continue

        print(
            f"Loading local S1 "
            f"{band}: {path.name}"
        )

        with MemoryFile(
            path.read_bytes()
        ) as memfile:

            with memfile.open() as src:

                image = src.read(1)

                bands[band] = image

                print(
                    f"Loaded local {band}: "
                    f"shape={image.shape}, "
                    f"dtype={image.dtype}"
                )

    missing = (
        EXPECTED_S1_BANDS
        - bands.keys()
    )

    if missing:

        raise RuntimeError(
            f"Local S1 patch is missing "
            f"bands: {sorted(missing)}"
        )

    return bands


# ============================================================
# LOAD S1 PATCH
# ============================================================

def load_s1_bands(
    s1_name: str,
    use_cache: bool = True,
    allow_remote_fallback: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Load VV and VH for one BigEarthNet-S1 patch.

    Priority:

    1. RAM cache
    2. Local extracted S1 patch
    3. Optional remote Zenodo fallback

    The remote fallback is disabled by default because
    sequentially scanning the 54 GB compressed archive
    is not suitable for normal query-time use.
    """

    # --------------------------------------------------------
    # RAM CACHE
    # --------------------------------------------------------

    if (
        use_cache
        and s1_name in S1_PATCH_CACHE
    ):

        print("=" * 60)
        print("S1 RAM CACHE HIT")
        print("=" * 60)

        print(
            "\nPatch:"
        )

        print(s1_name)

        return S1_PATCH_CACHE[
            s1_name
        ]

    # --------------------------------------------------------
    # LOCAL PATCH
    # --------------------------------------------------------

    local_patch_dir = (
        get_local_patch_dir(
            s1_name
        )
    )

    if local_patch_dir.exists():

        print("=" * 60)
        print("S1 LOCAL CACHE HIT")
        print("=" * 60)

        print(
            "\nPatch:"
        )

        print(s1_name)

        print(
            "\nDirectory:"
        )

        print(local_patch_dir)

        bands = load_local_s1_bands(
            s1_name
        )

        if use_cache:

            S1_PATCH_CACHE[
                s1_name
            ] = bands

            print(
                "\nS1 patch stored in RAM cache."
            )

        return bands

    # --------------------------------------------------------
    # REMOTE FALLBACK
    # --------------------------------------------------------

    if not allow_remote_fallback:

        raise FileNotFoundError(
            "S1 patch is not available "
            "in the local cache.\n\n"
            f"Expected:\n{local_patch_dir}\n\n"
            "Remote sequential fallback is "
            "disabled by default."
        )

    # --------------------------------------------------------
    # ORIGINAL REMOTE STREAM
    # --------------------------------------------------------

    target_root = make_target_root(
        s1_name
    )

    print("=" * 60)
    print("REMOTE BIGEARTHNET S1 PATCH")
    print("=" * 60)

    print(
        "\nTarget:"
    )

    print(s1_name)

    print(
        "\nTarget TAR directory:"
    )

    print(target_root)

    raw_reader = HTTPRangeReader(
        ZENODO_S1_URL,
        chunk_size=8 * 1024 * 1024,
    )

    zstd_reader = (
        zstd.ZstdDecompressor()
        .stream_reader(raw_reader)
    )

    bands: Dict[
        str,
        np.ndarray
    ] = {}

    with tarfile.open(
        fileobj=zstd_reader,
        mode="r|",
    ) as tar:

        scanned_members = 0

        for member in tar:

            scanned_members += 1

            if (
                scanned_members % 10000
                == 0
            ):

                print(
                    f"Scanned TAR members: "
                    f"{scanned_members}"
                )

            if not member.name.startswith(
                target_root
            ):
                continue

            if not member.name.lower().endswith(
                ".tif"
            ):
                continue

            band = band_name_from_path(
                member.name
            )

            if (
                band
                not in EXPECTED_S1_BANDS
            ):
                continue

            print(
                f"\nFOUND: {band}"
                f" | TAR size={member.size}"
            )

            extracted = tar.extractfile(
                member
            )

            if extracted is None:
                continue

            tif_bytes = (
                extracted.read()
            )

            with MemoryFile(
                tif_bytes
            ) as memfile:

                with memfile.open() as src:

                    image = src.read(1)

                    bands[band] = image

                    print(
                        f"Loaded {band}: "
                        f"shape={image.shape}, "
                        f"dtype={image.dtype}"
                    )

            if EXPECTED_S1_BANDS.issubset(
                bands.keys()
            ):

                print(
                    "\nAll S1 bands loaded."
                )

                break

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if not bands:

        raise RuntimeError(
            f"No S1 bands found for "
            f"{s1_name}"
        )

    missing = (
        EXPECTED_S1_BANDS
        - bands.keys()
    )

    if missing:

        raise RuntimeError(
            f"S1 patch {s1_name} "
            f"is missing bands: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # RAM CACHE
    # --------------------------------------------------------

    if use_cache:

        S1_PATCH_CACHE[
            s1_name
        ] = bands

        print(
            "\nS1 patch stored in RAM cache."
        )

    return bands


# ============================================================
# CACHE UTILITIES
# ============================================================

def cache_info():

    print("=" * 60)
    print("S1 CACHE")
    print("=" * 60)

    print(
        "\nRAM cached patches:",
        len(S1_PATCH_CACHE),
    )

    for s1_name in S1_PATCH_CACHE:

        print(
            f"- {s1_name}"
        )

    print(
        "\nLocal S1 patches:"
    )

    if not LOCAL_S1_CACHE_ROOT.exists():

        print(
            "- none"
        )

        return

    found = False

    for path in sorted(
        LOCAL_S1_CACHE_ROOT.iterdir()
    ):

        if path.is_dir():

            found = True

            print(
                f"- {path.name}"
            )

    if not found:
        print(
            "- none"
        )


def clear_cache():

    S1_PATCH_CACHE.clear()

    print(
        "S1 RAM cache cleared."
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        raise SystemExit(
            "Usage:\n"
            "python -m app.vlm.bigearthnet.remote_s1 "
            "<s1_name>"
        )

    s1_name = sys.argv[1]

    bands = load_s1_bands(
        s1_name
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "S1 PATCH LOADED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )

    print(
        "\nLoaded bands:",
        sorted(bands.keys()),
    )

    for name, image in sorted(
        bands.items()
    ):

        print(
            f"{name}: "
            f"shape={image.shape}, "
            f"dtype={image.dtype}, "
            f"min={image.min()}, "
            f"max={image.max()}"
        )