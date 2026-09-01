import requests

try:
    response = requests.post("http://127.0.0.1:8000/api/query", json={"query": "Compare urban/built-up change between 2022 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]"})
    print(response.json())
except Exception as e:
    print(e)
