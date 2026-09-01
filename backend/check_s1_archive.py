import tarfile
from huggingface_hub import hf_hub_download

TARGET = "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"

archive_path = hf_hub_download(
    repo_id="seosiju/BigEarthNet-S1",
    filename="BigEarthNet-S1_AT_pure.tar",
    repo_type="dataset",
)

print("=" * 70)
print("CHECKING LOCAL S1 ARCHIVE")
print("=" * 70)

print("\nArchive:")
print(archive_path)

found = []

with tarfile.open(archive_path, "r:") as tar:
    for member in tar:
        if TARGET in member.name:
            found.append(member.name)

print("\nMatching files:")

for name in found:
    print("-", name)

print("\nTotal matches:", len(found))