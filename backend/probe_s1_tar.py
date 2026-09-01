import io
import tarfile
import requests


URL = (
    "https://huggingface.co/datasets/"
    "seosiju/BigEarthNet-S1/resolve/main/"
    "BigEarthNet-S1_AT_pure.tar"
)

TARGET = (
    "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"
)

CHUNK_SIZE = 8 * 1024 * 1024


print("=" * 70)
print("S1 TAR RANGE PROBE")
print("=" * 70)

print("\nRequesting first 8 MB only...")

response = requests.get(
    URL,
    headers={
        "Range": f"bytes=0-{CHUNK_SIZE - 1}"
    },
    timeout=60,
)

response.raise_for_status()

print("HTTP status:", response.status_code)
print("Content-Range:", response.headers.get("Content-Range"))
print("Bytes received:", len(response.content))

buffer = io.BytesIO(
    response.content
)

print("\nReading TAR headers found inside first chunk...")

found = False
count = 0

with tarfile.open(
    fileobj=buffer,
    mode="r:",
) as tar:

    for member in tar:

        count += 1

        print(
            f"{count}: {member.name}"
        )

        if TARGET in member.name:
            found = True

        if count >= 30:
            break

print("\n" + "=" * 70)

if found:
    print("TARGET FOUND IN FIRST 8 MB")
else:
    print("TARGET NOT FOUND IN FIRST 8 MB")

print("=" * 70)