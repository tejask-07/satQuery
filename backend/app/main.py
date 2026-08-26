from fastapi import FastAPI
from app.api.routes_query import router as query_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SatQuery AI API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "satquery-api",
    }