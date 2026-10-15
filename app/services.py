"""Shared helpers for persisting a prediction as a Scan."""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from . import models
from .config import settings
from .detector import PredictionResult

# Suffixes we are willing to store/serve.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Magic-byte signatures, used because multipart Content-Type headers from
# clients (e.g. `package:http` sends application/octet-stream) are unreliable.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"BM", ".bmp"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
)


def sniff_image_suffix(data: bytes) -> str | None:
    """Return a file suffix if the bytes look like a known image, else None."""
    for signature, suffix in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return suffix
    # WEBP is "RIFF????WEBP".
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    # HEIC/HEIF is an ISO-BMFF box: "????ftyp" with a heic/heif/mif1 brand.
    if len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12] in {
        b"heic",
        b"heix",
        b"hevc",
        b"heim",
        b"heis",
        b"heif",
        b"mif1",
        b"msf1",
    }:
        return ".heic"
    return None


def looks_like_image(data: bytes) -> bool:
    """True when the bytes carry a recognizable image signature."""
    return sniff_image_suffix(data) is not None


def save_upload_image(data: bytes, filename_hint: str | None) -> str | None:
    """Persist the raw image bytes and return the stored filename."""
    if not settings.save_uploads:
        return None

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    # Prefer the sniffed type (authoritative) and fall back to the filename.
    suffix = sniff_image_suffix(data)
    if suffix is None:
        hint = Path(filename_hint or "").suffix.lower()
        suffix = hint if hint in IMAGE_SUFFIXES else ".jpg"
    filename = f"{uuid.uuid4().hex}{suffix}"
    (settings.upload_dir / filename).write_bytes(data)
    return filename


def persist_prediction(
    db: Session,
    prediction: PredictionResult,
    *,
    source: str,
    device_id: str | None,
    note: str | None,
    image_filename: str | None,
) -> models.Scan:
    """Create a Scan + its Detections from a prediction and commit."""
    detections = prediction.detections
    fruits = [d for d in detections if d.is_fruit]

    scan = models.Scan(
        source=source or "upload",
        device_id=device_id,
        note=note,
        image_filename=image_filename,
        image_url=f"/media/{image_filename}" if image_filename else None,
        image_width=prediction.width,
        image_height=prediction.height,
        model_name=prediction.model_name,
        latency_ms=prediction.latency_ms,
        detection_count=len(detections),
        fruit_count=len(fruits),
        max_confidence=max((d.confidence for d in detections), default=None),
    )

    for d in detections:
        scan.detections.append(
            models.Detection(
                label=d.label,
                display_name=d.display_name,
                category=d.category,
                is_fruit=d.is_fruit,
                confidence=d.confidence,
                class_id=d.class_id,
                x1=d.x1,
                y1=d.y1,
                x2=d.x2,
                y2=d.y2,
            )
        )

    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan
