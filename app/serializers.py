"""Convert ORM rows into API payloads (adds derived fields like percentages)."""

from __future__ import annotations

from datetime import datetime, timezone

from .models import Detection, Scan
from .schemas import DetectionOut, ScanDetail, ScanOut


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite drops timezone info; re-attach UTC so clients can localise."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def detection_to_out(det: Detection) -> DetectionOut:
    return DetectionOut(
        id=det.id,
        label=det.label,
        display_name=det.display_name,
        category=det.category,
        is_fruit=det.is_fruit,
        confidence=round(det.confidence, 4),
        confidence_percent=round(det.confidence * 100, 1),
        class_id=det.class_id,
        box={
            "x1": round(det.x1, 2),
            "y1": round(det.y1, 2),
            "x2": round(det.x2, 2),
            "y2": round(det.y2, 2),
        },
    )


def _scan_fields(scan: Scan) -> dict:
    return {
        "id": scan.id,
        "created_at": _as_utc(scan.created_at),
        "source": scan.source,
        "device_id": scan.device_id,
        "note": scan.note,
        "image_url": scan.image_url,
        "image_width": scan.image_width,
        "image_height": scan.image_height,
        "model_name": scan.model_name,
        "latency_ms": scan.latency_ms,
        "detection_count": scan.detection_count,
        "fruit_count": scan.fruit_count,
        "max_confidence": scan.max_confidence,
    }


def scan_to_out(scan: Scan) -> ScanOut:
    return ScanOut(**_scan_fields(scan))


def scan_to_detail(scan: Scan) -> ScanDetail:
    return ScanDetail(
        **_scan_fields(scan),
        detections=[detection_to_out(d) for d in scan.detections],
    )
