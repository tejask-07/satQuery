from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.api.routes_query import router as query_router
from app.api.routes_analysis import router as analysis_router
from app.api.routes_benchmark import router as benchmark_router


app = FastAPI(
    title="SatQuery AI API",
    version="0.1.0",
)


# ============================================================
# VISUALIZATION OUTPUTS
# ============================================================

VISUALIZATION_DIR = (
    Path(__file__).resolve().parent
    / "evidence"
    / "visualizations"
)

VISUALIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)




app.mount(
    "/visualizations",
    StaticFiles(
        directory=str(VISUALIZATION_DIR),
    ),
    name="visualizations",
)


# ============================================================
# DEBUG VISUALIZATION CHECK
# ============================================================

@app.get("/debug/visualization")
def debug_visualization():
    files = list(VISUALIZATION_DIR.glob("*.png"))

    return {
        "directory": str(VISUALIZATION_DIR),
        "exists": VISUALIZATION_DIR.exists(),
        "files": [file.name for file in files],
        "count": len(files),
    }


import os

# ============================================================
# CORS
# ============================================================

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [
    origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()
] if allowed_origins_env else [
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# VALIDATION EXCEPTION HANDLER
# ============================================================

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Format validation errors into structured payloads while preserving
    backward compatibility with string 'detail' assertions.
    """
    if isinstance(exc.detail, dict) and "error_code" in exc.detail:
        body = dict(exc.detail)
        if "detail" not in body or not isinstance(body["detail"], str):
            body["detail"] = body.get("message", "Validation error")
        return JSONResponse(status_code=exc.status_code, content=body)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# ============================================================
# ROUTES
# ============================================================

app.include_router(query_router)
app.include_router(analysis_router)
app.include_router(benchmark_router)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "satquery-api",
    }