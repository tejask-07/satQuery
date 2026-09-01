import binascii

import requests
import zstandard as zstd


ZENODO_S1_URL = (
    "https://zenodo.org/api/records/10891137"
    "/files/BigEarthNet-S1.tar.zst/content"
)


def main():

    print("=" * 60)
    print("S1 ZSTANDARD INSPECTION")
    print("=" * 60)

    response = requests.get(
        ZENODO_S1_URL,
        headers={
            "Range": "bytes=0-1048575"
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.content

    print(
        "HTTP STATUS:",
        response.status_code,
    )

    print(
        "CONTENT-RANGE:",
        response.headers.get(
            "Content-Range"
        ),
    )

    print(
        "BYTES RECEIVED:",
        len(data),
    )

    print(
        "\nZSTD MAGIC:",
        binascii.hexlify(
            data[:4]
        ).decode(),
    )

    print(
        "\nZSTD VERSION:",
        zstd.ZstdCompressor
    )

    # Try creating a streaming decompressor.
    try:

        decompressor = zstd.ZstdDecompressor()

        reader = decompressor.stream_reader(
            data
        )

        print(
            "\nStreaming decompressor:"
            " CREATED"
        )

        # Read only a small amount.
        sample = reader.read(1024)

        print(
            "Decompressed bytes:",
            len(sample),
        )

        print(
            "First decompressed bytes:"
        )

        print(
            binascii.hexlify(
                sample[:32]
            ).decode()
        )

        reader.close()

    except Exception as exc:

        print(
            "\nDecompression test failed:"
        )

        print(
            type(exc).__name__,
            str(exc),
        )


if __name__ == "__main__":
    main()