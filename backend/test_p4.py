import urllib.request
import json
import time

query = "Compare urban/built-up change between 2021 and 2024 for AOI [151.195, -33.885, 151.225, -33.855]"
aoi = {
    "type": "Polygon",
    "coordinates": [[[151.195, -33.885], [151.225, -33.885], [151.225, -33.855], [151.195, -33.855], [151.195, -33.885]]]
}

data = json.dumps({'query': query, 'aoi': aoi}).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:8000/api/query', data=data, headers={'Content-Type': 'application/json'})

start_time = time.time()
print("Sending request...")
response_bytes = urllib.request.urlopen(req).read()
resp = json.loads(response_bytes)

print(f"Status: {resp.get('status')}")
print(f"Answer: {resp.get('answer')}")
print(f"Execution Trace: {resp.get('execution_trace')}")
