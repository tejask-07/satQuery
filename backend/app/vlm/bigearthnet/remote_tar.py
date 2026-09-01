import io
import sys
import tarfile
from typing import Dict

import numpy as np
import requests
import zstandard as zstd
from rasterio.io import MemoryFile


ZENODO_S2_URL = (
    "https://zenodo.org/api/records/10891137"
    "/files/BigEarthNet-S2.tar.zst/content"
)


EXPECTED_BANDS = {
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
}


# In-memory cache.
# This survives only while the current Python process is running.
PATCH_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


class HTTPRangeReader(io.RawIOBase):
    """
    Sequential remote reader using HTTP Range requests.

    The full Zenodo archive is never saved locally.
    Only the current chunk is held in memory.
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
        end = start + self.chunk_size - 1

        response = requests.get(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}"
            },
            timeout=120,
        )

        response.raise_for_status()

        if response.status_code != 206:
            raise RuntimeError(
                f"Expected HTTP 206, got {response.status_code}"
            )

        content_range = response.headers.get(
            "Content-Range"
        )

        print(
            f"HTTP Range: {content_range}"
        )

        self.buffer = response.content
        self.buffer_position = 0

        if content_range and "/" in content_range:
            try:
                self.total_size = int(
                    content_range.split("/")[-1]
                )
            except ValueError:
                pass

        if not self.buffer:
            raise EOFError(
                "Zenodo returned an empty response."
            )

    def read(self, size=-1):
        if size == 0:
            return b""

        output = bytearray()

        while size < 0 or len(output) < size:

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
                    size - len(output)
                )
                take = min(
                    available,
                    remaining,
                )

            start = self.buffer_position
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


def band_name_from_path(path: str) -> str:
    """
    Extract B01/B02/... from a BigEarthNet TIFF path.
    """

    filename = path.rsplit("/", 1)[-1]

    band = filename.rsplit("_", 1)[-1]

    return band.replace(".tif", "")


def make_target_root(patch_id: str) -> str:
    """
    Build the directory path inside the Zenodo TAR
    for a given BigEarthNet Sentinel-2 patch.
    """

    parts = patch_id.split("_")

    if len(parts) < 2:
        raise ValueError(
            f"Invalid BigEarthNet patch_id: {patch_id}"
        )

    granule_name = "_".join(
        parts[:-2]
    )

    return (
        "BigEarthNet-S2/"
        f"{granule_name}/"
        f"{patch_id}/"
    )


def load_patch_bands(
    patch_id: str,
    use_cache: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load all Sentinel-2 TIFF bands for the supplied patch.

    The archive remains remote.
    TIFF pixel data is loaded into RAM only.

    If use_cache=True:
        return an already-loaded patch from RAM.
    """

    # --------------------------------------------------------
    # CACHE CHECK
    # --------------------------------------------------------

    if (
        use_cache
        and patch_id in PATCH_CACHE
    ):
        print("=" * 60)
        print("S2 PATCH CACHE HIT")
        print("=" * 60)

        print("\nPatch:")
        print(patch_id)

        return PATCH_CACHE[patch_id]

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target_root = make_target_root(
        patch_id
    )

    print("=" * 60)
    print("REMOTE BIGEARTHNET S2 PATCH")
    print("=" * 60)

    print("\nTarget:")
    print(patch_id)

    print("\nTarget TAR directory:")
    print(target_root)

    # --------------------------------------------------------
    # REMOTE STREAM
    # --------------------------------------------------------

    raw_reader = HTTPRangeReader(
        ZENODO_S2_URL,
        chunk_size=8 * 1024 * 1024,
    )

    zstd_reader = (
        zstd.ZstdDecompressor()
        .stream_reader(raw_reader)
    )

    bands: Dict[str, np.ndarray] = {}

    # --------------------------------------------------------
    # TAR STREAM
    # --------------------------------------------------------

    with tarfile.open(
        fileobj=zstd_reader,
        mode="r|",
    ) as tar:

        for member in tar:

            # Ignore other patches.
            if not member.name.startswith(
                target_root
            ):
                continue

            # We only need TIFF bands.
            if not member.name.lower().endswith(
                ".tif"
            ):
                continue

            band = band_name_from_path(
                member.name
            )

            print(
                f"\nFOUND: {band}"
                f" | TAR size={member.size}"
            )

            extracted = tar.extractfile(
                member
            )

            if extracted is None:
                continue

            tif_bytes = extracted.read()

            # Read the TIFF entirely in memory.
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

            # Stop immediately after all 12 bands.
            if EXPECTED_BANDS.issubset(
                bands.keys()
            ):
                print(
                    "\nAll S2 bands loaded."
                )

                print(
                    "Stopping remote stream."
                )

                break

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not bands:
        raise RuntimeError(
            f"No S2 bands found for patch "
            f"{patch_id}"
        )

    missing = (
        EXPECTED_BANDS - bands.keys()
    )

    if missing:
        print(
            "\nWARNING: Missing bands:",
            sorted(missing),
        )

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if use_cache:

        PATCH_CACHE[patch_id] = bands

        print(
            "\nPatch stored in RAM cache."
        )

    return bands


def cache_info():
    """
    Display patches currently stored in RAM.
    """

    print("=" * 60)
    print("S2 CACHE")
    print("=" * 60)

    if not PATCH_CACHE:
        print("\nCache is empty.")
        return

    print(
        f"\nCached patches: "
        f"{len(PATCH_CACHE)}"
    )

    for patch_id in PATCH_CACHE:
        print(
            f"- {patch_id}"
        )


def clear_cache():
    """
    Clear the RAM cache.
    """

    PATCH_CACHE.clear()

    print(
        "S2 patch cache cleared."
    )


# ------------------------------------------------------------
# COMMAND-LINE TEST
# ------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "python -m app.vlm.bigearthnet.remote_tar "
            "<patch_id>"
        )

    patch_id = sys.argv[1]

    bands = load_patch_bands(
        patch_id
    )

    print("\n" + "=" * 60)
    print("PATCH LOADED SUCCESSFULLY")
    print("=" * 60)

    print("\nLoaded bands:")
    print(sorted(bands.keys()))

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