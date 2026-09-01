import requests
from pathlib import Path


RECORD_ID = "10891137"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"

OUTPUT_DIR = Path("data/bigearthnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_record():
    response = requests.get(
        API_URL,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def find_file(record, filename):
    for file in record["files"]:
        if file["key"] == filename:
            return file

    raise RuntimeError(f"File not found: {filename}")


def download_small_file(url, output_path):
    print(f"\nDownloading:")
    print(url)

    response = requests.get(
        url,
        timeout=120,
    )

    response.raise_for_status()

    output_path.write_bytes(response.content)

    print(f"Saved to: {output_path}")
    print(f"Size: {len(response.content):,} bytes")


if __name__ == "__main__":

    record = get_record()

    metadata_file = find_file(
        record,
        "metadata.parquet"
    )

    metadata_url = metadata_file["links"]["self"]

    output_path = (
        OUTPUT_DIR /
        "metadata.parquet"
    )

    download_small_file(
        metadata_url,
        output_path
    )