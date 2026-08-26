from fastapi import FastAPI

from app.api.routes_query import router as query_router

app = FastAPI(
    title="SatQuery AI API",
    version="0.1.0",
)

app.include_router(query_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "satquery-api",
    }