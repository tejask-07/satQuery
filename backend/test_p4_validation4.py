import json
from unittest.mock import patch
from app.api.routes_query import process_query
from app.schemas.query import QueryRequest
from app.vlm.model import VLM

def intercept_generate(self, image=None, question="", evidence=None, images=None):
    print("--- EVIDENCE START ---")
    print(evidence)
    print("--- EVIDENCE END ---")
    return "MOCKED VLM RESPONSE"

if __name__ == "__main__":
    req = QueryRequest(
        query="Compare urban/built-up change between 2021 and 2025 for AOI [151.195, -33.885, 151.225, -33.855]",
        aoi={
            "type": "Polygon",
            "coordinates": [[[151.195, -33.885], [151.225, -33.885], [151.225, -33.855], [151.195, -33.855], [151.195, -33.885]]]
        }
    )
    with patch.object(VLM, 'generate', new=intercept_generate):
        process_query(req)
