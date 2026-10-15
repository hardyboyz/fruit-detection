"""Detection endpoints: upload an image, get fruits back."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..detector import DetectorUnavailable, detector
from ..schemas import DetectionOut, PreviewOut, ScanDetail
from ..serializers import scan_to_detail
from ..services import looks_like_image, persist_prediction, save_upload_image

router = APIRouter(prefix="/api", tags=["detection"])

# Content types that carry no useful information and must be validated by
# inspecting the payload instead (e.g. `package:http` sends octet-stream).
_OPAQUE_CONTENT_TYPES = {
    "",
    "application/octet-stream",
    "binary/octet-stream",
    "application/binary",
}


def _validate_upload(file: UploadFile, data: bytes) -> None:
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (max {settings.max_upload_mb} MB).",
        )

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type.startswith("image/"):
        return
    # Trust the bytes rather than a generic/opaque header from the client.
    if content_type in _OPAQUE_CONTENT_TYPES and looks_like_image(data):
        return
    raise HTTPException(
        status_code=415,
        detail=(
            f"Unsupported content type '{file.content_type or 'unknown'}'. "
            "Send an image (jpeg, png, webp, bmp or heic)."
        ),
    )


def _run(data: bytes, conf: float | None, iou: float | None):
    try:
        return detector.predict(data, conf=conf, iou=iou)
    except DetectorUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/detect", response_model=ScanDetail, summary="Detect fruits and save the scan")
async def detect_and_save(
    file: UploadFile = File(..., description="The image to scan."),
    conf: float | None = Form(None, ge=0.0, le=1.0, description="Confidence threshold."),
    iou: float | None = Form(None, ge=0.0, le=1.0, description="NMS IoU threshold."),
    source: str = Form("upload", description="camera | gallery | upload"),
    device_id: str | None = Form(None, description="Client device identifier."),
    note: str | None = Form(None),
    db: Session = Depends(get_db),
) -> ScanDetail:
    data = await file.read()
    _validate_upload(file, data)

    prediction = _run(data, conf, iou)
    image_filename = save_upload_image(data, file.filename)

    scan = persist_prediction(
        db,
        prediction,
        source=source,
        device_id=device_id,
        note=note,
        image_filename=image_filename,
    )
    return scan_to_detail(scan)


@router.post("/detect/preview", response_model=PreviewOut, summary="Detect fruits without saving")
async def detect_preview(
    file: UploadFile = File(...),
    conf: float | None = Form(None, ge=0.0, le=1.0),
    iou: float | None = Form(None, ge=0.0, le=1.0),
) -> PreviewOut:
    data = await file.read()
    _validate_upload(file, data)

    prediction = _run(data, conf, iou)
    return PreviewOut(
        width=prediction.width,
        height=prediction.height,
        latency_ms=prediction.latency_ms,
        model_name=prediction.model_name,
        detections=[DetectionOut(**d.as_dict()) for d in prediction.detections],
    )
