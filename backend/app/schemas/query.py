from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str

    # Optional user-provided images
    image_ids: List[str] = Field(default_factory=list)

    # Optional Area of Interest
    aoi: Optional[dict] = None


class QueryPlan(BaseModel):
    # What the user wants to do
    task: str

    # What the analysis is about
    target: Optional[str] = None

    # Temporal constraints
    time_start: Optional[str] = None
    time_end: Optional[str] = None

    # Satellite / sensing modalities
    modalities: List[str] = Field(default_factory=list)

    # Specific remote-sensing metric
    metric: Optional[str] = None

    # Desired direction of change
    direction: Literal[
        "increase",
        "decrease",
        "both",
        "unknown",
    ] = "unknown"

    # Operations that need to be performed
    analysis: List[str] = Field(default_factory=list)

    # What the frontend should display
    output: List[str] = Field(default_factory=list)