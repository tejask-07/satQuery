from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    status: str
    answer: Optional[str] = None
    confidence: Optional[float] = None
    plan: Optional[Dict[str, Any]] = None
    statistics: Dict[str, Any] = Field(default_factory=dict)
    layers: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    execution_trace: List[str] = Field(default_factory=list)