import requests
from typing import Any, Dict


DEFAULT_P2_URL = (
    "http://127.0.0.1:8000"
)


def query_p2(
    query: str,
    base_url: str = DEFAULT_P2_URL,
) -> Dict[str, Any]:
    """
    Call Person 2's existing FastAPI endpoint.
    """

    url = (
        f"{base_url.rstrip('/')}"
        "/api/query"
    )

    response = requests.post(
        url,
        json={
            "query": query,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") != "success":
        raise RuntimeError(
            "P2 API returned a non-success response:\n"
            f"{data}"
        )

    return data