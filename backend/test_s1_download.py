from huggingface_hub import hf_hub_download

REPO_ID = "seosiju/BigEarthNet-S1"
FILENAME = "BigEarthNet-S1_AT_pure.tar"

print("=" * 60)
print("S1 AUSTRIA ARCHIVE DOWNLOAD TEST")
print("=" * 60)

print("\nDownloading/caching:")
print(FILENAME)

path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    repo_type="dataset",
)

print("\nDownloaded/cached file:")
print(path)