"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    label: str
    display_name: str
    category: str
    is_fruit: bool
    confidence: float = Field(..., description="0.0 - 1.0")
    confidence_percent: float = Field(..., description="0 - 100, rounded to 1 decimal")
    class_id: int | None = None
    box: dict[str, float]


class ScanOut(BaseModel):
    """Lightweight scan summary used by the history table."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    source: str
    device_id: str | None = None
    note: str | None = None
    image_url: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    model_name: str | None = None
    latency_ms: float | None = None
    detection_count: int
    fruit_count: int
    max_confidence: float | None = None


class ScanDetail(ScanOut):
    detections: list[DetectionOut] = []


class PreviewOut(BaseModel):
    """Detection result that is *not* persisted to the database."""

    width: int
    height: int
    latency_ms: float
    model_name: str
    detections: list[DetectionOut] = []


class ScanListOut(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ScanOut]


class FruitStat(BaseModel):
    name: str
    count: int
    avg_confidence: float
    max_confidence: float


class StatsOut(BaseModel):
    total_scans: int
    total_detections: int
    total_fruits: int
    avg_confidence: float | None = None
    top_fruits: list[FruitStat] = []


class ClassInfo(BaseModel):
    name: str
    category: str
    is_fruit: bool


class ModelStatus(BaseModel):
    loaded: bool
    model_name: str | None = None
    weights_path: str | None = None
    is_custom_fruit_model: bool = False
    class_count: int = 0
    fruit_class_count: int = 0
    device: str | None = None
    error: str | None = None
    # Multi-model ensemble info (see app/detector.py).
    ensemble_active: bool = False
    secondary_model_name: str | None = None
    secondary_weights_path: str | None = None
    classifier_models: list[str] = []


class HealthOut(BaseModel):
    status: str
    version: str
    database: str
    model: ModelStatus
