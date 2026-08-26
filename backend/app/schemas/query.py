from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    image_ids: List[str] = Field(default_factory=list)
    aoi: Optional[dict] = None


class QueryPlan(BaseModel):
    task: str
    target: Optional[str] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    modalities: List[str] = Field(default_factory=list)
    analysis: List[str] = Field(default_factory=list)
    output: List[str] = Field(default_factory=list)