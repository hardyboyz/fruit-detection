"""Scan history endpoints — the data behind the mobile app's table."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Detection, Scan
from ..schemas import FruitStat, ScanDetail, ScanListOut, StatsOut
from ..serializers import scan_to_detail, scan_to_out

router = APIRouter(prefix="/api/scans", tags=["scans"])


@router.get("", response_model=ScanListOut, summary="List scans (newest first)")
def list_scans(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    device_id: str | None = Query(None),
    fruits_only: bool = Query(False, description="Only scans that found ≥1 fruit."),
    search: str | None = Query(None, description="Filter by fruit name."),
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
) -> ScanListOut:
    filters = []
    if device_id:
        filters.append(Scan.device_id == device_id)
    if fruits_only:
        filters.append(Scan.fruit_count > 0)
    if since:
        filters.append(Scan.created_at >= since)
    if until:
        filters.append(Scan.created_at <= until)
    if search:
        # Match the Indonesian display name *and* the raw model label, so both
        # "pisang" and "banana" find the same scan.
        filters.append(
            or_(
                Scan.detections.any(Detection.display_name.ilike(f"%{search}%")),
                Scan.detections.any(Detection.label.ilike(f"%{search}%")),
            )
        )

    total = db.scalar(select(func.count(Scan.id)).where(*filters)) or 0

    rows = (
        db.scalars(
            select(Scan)
            .where(*filters)
            .order_by(Scan.created_at.desc(), Scan.id.desc())
            .limit(limit)
            .offset(offset)
        )
        .unique()
        .all()
    )

    return ScanListOut(
        total=total,
        limit=limit,
        offset=offset,
        items=[scan_to_out(s) for s in rows],
    )


@router.get("/stats", response_model=StatsOut, summary="Aggregate statistics")
def stats(
    db: Session = Depends(get_db),
    device_id: str | None = Query(None),
) -> StatsOut:
    scan_filter = [Scan.device_id == device_id] if device_id else []
    det_filter = (
        [Detection.scan_id.in_(select(Scan.id).where(*scan_filter))]
        if scan_filter
        else []
    )

    total_scans = db.scalar(select(func.count(Scan.id)).where(*scan_filter)) or 0
    total_detections = db.scalar(select(func.count(Detection.id)).where(*det_filter)) or 0
    total_fruits = (
        db.scalar(
            select(func.count(Detection.id)).where(
                Detection.is_fruit.is_(True), *det_filter
            )
        )
        or 0
    )
    avg_conf = db.scalar(select(func.avg(Detection.confidence)).where(*det_filter))

    rows = db.execute(
        select(
            Detection.display_name,
            func.count(Detection.id).label("count"),
            func.avg(Detection.confidence).label("avg_conf"),
            func.max(Detection.confidence).label("max_conf"),
        )
        .where(Detection.is_fruit.is_(True), *det_filter)
        .group_by(Detection.display_name)
        .order_by(func.count(Detection.id).desc())
        .limit(10)
    ).all()

    return StatsOut(
        total_scans=total_scans,
        total_detections=total_detections,
        total_fruits=total_fruits,
        avg_confidence=round(avg_conf, 4) if avg_conf is not None else None,
        top_fruits=[
            FruitStat(
                name=r.display_name,
                count=r.count,
                avg_confidence=round(r.avg_conf, 4),
                max_confidence=round(r.max_conf, 4),
            )
            for r in rows
        ],
    )


@router.get("/{scan_id}", response_model=ScanDetail, summary="Get one scan")
def get_scan(scan_id: int, db: Session = Depends(get_db)) -> ScanDetail:
    scan = db.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found.")
    return scan_to_detail(scan)


@router.delete(
    "/{scan_id}",
    status_code=204,
    response_class=Response,
    summary="Delete one scan",
)
def delete_scan(scan_id: int, db: Session = Depends(get_db)) -> Response:
    scan = db.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found.")
    db.delete(scan)
    db.commit()
    return Response(status_code=204)


@router.delete("", summary="Delete all scans (optionally for one device)")
def clear_scans(
    db: Session = Depends(get_db),
    device_id: str | None = Query(None),
) -> dict[str, int]:
    stmt = select(Scan)
    if device_id:
        stmt = stmt.where(Scan.device_id == device_id)
    scans = db.scalars(stmt).unique().all()
    count = len(scans)
    for scan in scans:
        db.delete(scan)
    db.commit()
    return {"deleted": count}
