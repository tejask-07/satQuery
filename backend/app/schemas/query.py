from typing import List, Optional, Literal, Union

from pydantic import BaseModel, Field, model_validator


class QueryRequest(BaseModel):
    query: str

    # Optional user-provided images
    image_ids: List[str] = Field(default_factory=list)

    # Optional Area of Interest (GeoJSON dict or bbox list)
    aoi: Optional[Union[dict, list]] = None

    # Optional explicit Optical and SAR multimodal image identifiers
    optical_image_id: Optional[str] = None
    sar_image_id: Optional[str] = None

    # Optional temporal parameters (Step 14)
    time_start: Optional[str] = None
    time_end: Optional[str] = None



class QueryPlan(BaseModel):
    # What the user wants to do (backward-compatible task name)
    task: str

    # High-level intent of the analysis
    intent: Optional[str] = None

    # Area of Interest selected by the user
    aoi: Optional[Union[dict, list]] = None

    # What the analysis is about (single target or summary target)
    target: Optional[str] = None

    # Specific targets for multi-target or single-target analysis
    targets: List[str] = Field(default_factory=list)

    # Land-cover transition endpoints (when intent is land_cover_transition)
    source: Optional[str] = None
    destination: Optional[str] = None

    # Temporal constraints
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    temporal_mode: Optional[str] = "bi_temporal"

    # Satellite / sensing modalities
    modalities: List[str] = Field(default_factory=list)

    # Specific remote-sensing metric (e.g., NDVI, NDWI, NDBI)
    metric: Optional[str] = None

    # Explicitly requested metric in query text (e.g., "NDVI", "NDWI", "NDBI")
    explicit_metric: Optional[str] = None

    # Primary remote sensing indicators (e.g., ["NDBI"], ["NDVI", "NDWI", "NDBI"])
    primary_indicators: List[str] = Field(default_factory=list)

    # Supporting contextual / scientific indicators
    supporting_indicators: List[str] = Field(default_factory=list)

    # Evidence requirements to be validated
    evidence_requirements: List[str] = Field(default_factory=list)

    # Desired direction of change
    direction: Literal[
        "increase",
        "decrease",
        "both",
        "unknown",
    ] = "unknown"

    # Operations that need to be performed
    analysis: List[str] = Field(default_factory=list)

    # What the frontend should display (legacy field name)
    output: List[str] = Field(default_factory=list)

    # Desired deliverables / presentation outputs
    outputs: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_legacy_and_extended_fields(self):
        # Sync target and targets
        if self.target and not self.targets:
            self.targets = [self.target]
        elif not self.target and len(self.targets) == 1:
            self.target = self.targets[0]

        # Sync intent and task
        if not self.intent:
            self.intent = self.task

        # Sync output and outputs
        if self.outputs and not self.output:
            self.output = [o for o in self.outputs if o != "confidence"]
        elif self.output and not self.outputs:
            self.outputs = list(self.output)
            if "confidence" not in self.outputs:
                self.outputs.append("confidence")

        return self