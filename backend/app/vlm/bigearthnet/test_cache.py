from app.vlm.bigearthnet.remote_tar import (
    load_patch_bands,
    cache_info,
)


PATCH_ID = (
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"
)


def main():

    print("=" * 60)
    print("S2 CACHE TEST")
    print("=" * 60)

    print("\nFIRST REQUEST")
    bands1 = load_patch_bands(
        PATCH_ID
    )

    print(
        "\nFirst request loaded:",
        sorted(bands1.keys())
    )

    print("\nSECOND REQUEST")
    bands2 = load_patch_bands(
        PATCH_ID
    )

    print(
        "\nSecond request loaded:",
        sorted(bands2.keys())
    )

    print("\nCACHE INFORMATION")
    cache_info()

    print("\nSame object:")
    print(bands1 is bands2)


if __name__ == "__main__":
    main()