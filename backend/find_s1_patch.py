import requests


DATASET = "seosiju/BigEarthNet-S1"
CONFIG = "default"
SPLIT = "partial-train"

TARGET_S1 = (
    "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"
)

url = "https://datasets-server.huggingface.co/filter"

params = {
    "dataset": DATASET,
    "config": CONFIG,
    "split": SPLIT,
    "where": f'"s1_name"=\'{TARGET_S1}\'',
    "offset": 0,
    "length": 10,
}

print("=" * 70)
print("BIGEARTHNET S1 PARQUET LOOKUP")
print("=" * 70)

print("\nTarget:")
print(TARGET_S1)

print("\nQuerying Hugging Face Dataset Viewer...")

response = requests.get(
    url,
    params=params,
    timeout=60,
)

print("\nHTTP status:")
print(response.status_code)

response.raise_for_status()

data = response.json()

print("\nMatches:")

for item in data.get("rows", []):
    print(item)