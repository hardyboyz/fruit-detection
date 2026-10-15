"""SQLAlchemy ORM models.

A *Scan* is one uploaded image. Each *Detection* is one fruit (or object) that
YOLO found inside that image. Keeping them separate means a single photo of a
fruit bowl produces one Scan with many Detections.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    # "camera" / "gallery" / "upload" — whatever the client reports.
    source: Mapped[str] = mapped_column(String(32), default="upload")
    # Free-form client identifier so one device can filter its own history.
    device_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Original image metadata + where the file was stored (relative to upload_dir).
    image_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # How the inference was performed.
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Denormalised counters to make list queries cheap.
    detection_count: Mapped[int] = mapped_column(Integer, default=0)
    fruit_count: Mapped[int] = mapped_column(Integer, default=0)
    max_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="scan",
        cascade="all, delete-orphan",
        order_by="Detection.confidence.desc()",
        lazy="selectin",
    )


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_id: Mapped[int] = mapped_column(
        ForeignKey("scans.id", ondelete="CASCADE"), index=True
    )

    # Raw label produced by the model and its normalised display name.
    label: Mapped[str] = mapped_column(String(128), index=True)
    display_name: Mapped[str] = mapped_column(String(128), index=True)
    category: Mapped[str] = mapped_column(String(32), default="other")  # fruit/vegetable/other
    is_fruit: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    confidence: Mapped[float] = mapped_column(Float)  # 0.0 - 1.0
    class_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Bounding box in pixels of the *original* uploaded image.
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)

    scan: Mapped[Scan] = relationship(back_populates="detections")
