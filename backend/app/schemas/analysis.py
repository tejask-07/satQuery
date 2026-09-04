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
    visualization_url: Optional[str] = None
    classified_visualization_url: Optional[str] = None
    bounds: Optional[List[List[float]]] = None
    images: Optional[Dict[str, Any]] = None
    layer_package: Optional[Dict[str, Any]] = None
    multi_index_evidence: Optional[Dict[str, Any]] = None
    evidence_package: Optional[Dict[str, Any]] = None
    candidates: Optional[List[Dict[str, Any]]] = None
    candidate_package: Optional[Dict[str, Any]] = None
    interpretation: Optional[Dict[str, Any]] = None
    spatial_analysis: Optional[Dict[str, Any]] = None
    temporal_analysis: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None