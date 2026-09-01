import tarfile
from pathlib import Path


ARCHIVE_PATH = Path(
    r"C:\Users\sansk\.cache\huggingface\hub\datasets--seosiju--BigEarthNet-S1"
) / "snapshots"

TARGET = "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "s1_cache"
    / TARGET
)


def find_archive():
    matches = list(
        ARCHIVE_PATH.rglob(
            "BigEarthNet-S1_AT_pure.tar"
        )
    )

    if not matches:
        raise FileNotFoundError(
            "Cached BigEarthNet-S1_AT_pure.tar was not found."
        )

    return matches[0]


def main():
    print("=" * 70)
    print("EXTRACT LOCAL S1 PATCH")
    print("=" * 70)

    archive_path = find_archive()

    print("\nUsing LOCAL archive:")
    print(archive_path)

    print(
        "\nArchive size:",
        f"{archive_path.stat().st_size / (1024**3):.2f} GiB",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    found = []

    print("\nScanning local TAR...")

    with tarfile.open(
        archive_path,
        mode="r:",
    ) as tar:

        for member in tar:

            if TARGET not in member.name:
                continue

            if not member.name.lower().endswith(".tif"):
                continue

            filename = Path(member.name).name

            print(
                f"\nFOUND: {member.name}"
            )

            extracted = tar.extractfile(member)

            if extracted is None:
                continue

            output_path = OUTPUT_DIR / filename

            with open(
                output_path,
                "wb",
            ) as out:

                while True:
                    chunk = extracted.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    out.write(chunk)

            print(
                f"SAVED: {output_path}"
            )

            found.append(output_path)

            if len(found) == 2:
                break

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"\nFound {len(found)} TIFF files."
    )

    for path in found:
        print(
            f"- {path.name}"
        )

    if len(found) == 0:
        raise RuntimeError(
            f"Target patch not found: {TARGET}"
        )

    print(
        "\nOutput directory:"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()