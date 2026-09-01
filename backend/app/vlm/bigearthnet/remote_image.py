import io
from pathlib import Path

import requests
import rasterio
from rasterio.io import MemoryFile


ZENODO_RECORD_ID = "10891137"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def get_record():
    response = requests.get(
        ZENODO_API_URL,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_zenodo_file_url(filename: str) -> str:
    record = get_record()

    for file in record["files"]:
        if file["key"] == filename:
            return file["links"]["self"]

    raise FileNotFoundError(
        f"Zenodo file not found: {filename}"
    )


def read_range(url: str, start: int, end: int) -> bytes:
    response = requests.get(
        url,
        headers={
            "Range": f"bytes={start}-{end}"
        },
        timeout=60,
    )

    response.raise_for_status()

    if response.status_code != 206:
        raise RuntimeError(
            f"Expected HTTP 206, got {response.status_code}"
        )

    return response.content


def test_remote_access():
    """
    Verify that Zenodo allows partial HTTP access.
    Only 1 MB is requested.
    """

    url = get_zenodo_file_url(
        "BigEarthNet-S2.tar.zst"
    )

    print("S2 URL:")
    print(url)

    data = read_range(
        url,
        0,
        1024 * 1024 - 1,
    )

    print("\nHTTP Range test successful")
    print("Downloaded bytes:", len(data))


if __name__ == "__main__":
    test_remote_access()