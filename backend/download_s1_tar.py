from pathlib import Path
import requests


URL = (
    "https://huggingface.co/datasets/"
    "seosiju/BigEarthNet-S1/resolve/main/"
    "BigEarthNet-S1_AT_pure.tar"
)

OUTPUT = Path("BigEarthNet-S1_AT_pure.tar")
CHUNK_SIZE = 8 * 1024 * 1024


def main():
    existing = OUTPUT.stat().st_size if OUTPUT.exists() else 0

    headers = {}

    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming from {existing:,} bytes...")
    else:
        print("Starting download...")

    with requests.get(
        URL,
        headers=headers,
        stream=True,
        timeout=120,
    ) as response:

        response.raise_for_status()

        if existing > 0 and response.status_code != 206:
            print(
                "Server did not honor the resume request. "
                "Restarting cleanly."
            )
            existing = 0

        total_from_header = response.headers.get(
            "Content-Range"
        )

        total = None

        if total_from_header and "/" in total_from_header:
            total = int(total_from_header.split("/")[-1])
        elif response.headers.get("Content-Length"):
            total = existing + int(
                response.headers["Content-Length"]
            )

        mode = "ab" if existing > 0 else "wb"

        downloaded = existing

        with open(OUTPUT, mode) as f:
            for chunk in response.iter_content(
                chunk_size=CHUNK_SIZE
            ):
                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = (
                        downloaded / total * 100
                    )
                    print(
                        f"\r{downloaded / 1024**3:.2f} / "
                        f"{total / 1024**3:.2f} GiB "
                        f"({percent:.1f}%)",
                        end="",
                        flush=True,
                    )

    print()
    print("Download complete:")
    print(OUTPUT.resolve())
    print("Size:", OUTPUT.stat().st_size)


if __name__ == "__main__":
    main()