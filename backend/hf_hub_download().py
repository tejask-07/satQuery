import tarfile
from pathlib import Path


ARCHIVE = Path("BigEarthNet-S1_AT_pure.tar")

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
    print("EXTRACTING TARGET S1 PATCH")
    print("=" * 70)

    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"Archive not found: {ARCHIVE.resolve()}"
        )

    print("\nArchive:")
    print(ARCHIVE.resolve())

    print("\nTarget:")
    print(TARGET)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    found = []

    print("\nScanning local TAR...")

    with tarfile.open(
        ARCHIVE,
        mode="r:",
    ) as tar:

        for member in tar:

            if TARGET not in member.name:
                continue

            if not member.name.lower().endswith(".tif"):
                continue

            filename = Path(member.name).name

            print(f"\nFOUND: {member.name}")

            extracted = tar.extractfile(member)

            if extracted is None:
                print("Could not extract member.")
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

            print(f"SAVED: {output_path}")

            found.append(output_path)

            if len(found) == 2:
                break

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    if not found:
        raise RuntimeError(
            f"Target was not found: {TARGET}"
        )

    for path in found:
        print(
            f"{path.name} | "
            f"{path.stat().st_size:,} bytes"
        )

    print("\nPatch directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()