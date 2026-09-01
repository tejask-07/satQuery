# pyrefly: ignore [missing-import]
from huggingface_hub import hf_hub_download
import pandas as pd


REPO_ID = "seosiju/BigEarthNet-S1"
FILENAME = "metadata.parquet"


print("=" * 60)
print("BIGEARTHNET S1 MIRROR METADATA")
print("=" * 60)

print("\nDownloading metadata only...")

path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    repo_type="dataset",
)

print("\nMetadata path:")
print(path)

df = pd.read_parquet(path)

print("\nColumns:")
print(list(df.columns))

print("\nRows:")
print(len(df))

print("\nFirst rows:")
print(df.head().to_string())