import tarfile
from pathlib import Path


ARCHIVE = Path(
    "BigEarthNet-S1_AT_pure.tar"
)

TARGET = (
    "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"
)

OUTPUT_DIR = (
    Path("data")
    / "s1_cache"
    / TARGET
)


def main():

    print("=" * 70)
    print("LOCAL S1 PATCH EXTRACTION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Check local archive
    # ---------------------------------------------------------

    if not ARCHIVE.exists():

        raise FileNotFoundError(
            f"Archive not found:\n"
            f"{ARCHIVE.resolve()}"
        )

    print("\nUsing existing LOCAL archive:")
    print(ARCHIVE.resolve())

    print(
        "\nArchive size:",
        f"{ARCHIVE.stat().st_size / (1024**3):.2f} GiB"
    )

    print("\nTarget:")
    print(TARGET)

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    found = []

    # ---------------------------------------------------------
    # Open local TAR
    # ---------------------------------------------------------

    print("\nScanning LOCAL TAR...")
    print("NO DOWNLOAD WILL OCCUR.")

    with tarfile.open(
        ARCHIVE,
        mode="r:"
    ) as tar:

        for member in tar:

            if TARGET not in member.name:
                continue

            if not member.name.lower().endswith(".tif"):
                continue

            filename = Path(
                member.name
            ).name

            print(
                f"\nFOUND: {member.name}"
            )

            extracted = tar.extractfile(
                member
            )

            if extracted is None:
                print(
                    "Could not extract:",
                    member.name
                )
                continue

            output_path = (
                OUTPUT_DIR
                / filename
            )

            with open(
                output_path,
                "wb"
            ) as output_file:

                while True:

                    chunk = extracted.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    output_file.write(
                        chunk
                    )

            print(
                f"SAVED: {output_path}"
            )

            found.append(
                output_path
            )

            # We only need VV + VH.
            if len(found) >= 2:
                break

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXTRACTION RESULT")
    print("=" * 70)

    print(
        f"\nFiles extracted: {len(found)}"
    )

    for path in found:

        print(
            f"- {path.name}"
            f" | {path.stat().st_size:,} bytes"
        )

    if not found:

        raise RuntimeError(
            f"Target patch was not found:\n{TARGET}"
        )

    print(
        "\nPatch directory:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()