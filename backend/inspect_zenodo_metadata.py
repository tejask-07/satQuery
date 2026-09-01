import json
import requests

url = "https://zenodo.org/api/records/10891137"

response = requests.get(
    url,
    timeout=30,
)

response.raise_for_status()

data = response.json()

print(
    json.dumps(
        data.get("metadata", {}),
        indent=2,
    )
)