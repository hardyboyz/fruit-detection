"""Meta endpoints: health, model status, class list, model reload."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..detector import detector
from ..schemas import ClassInfo, HealthOut, ModelStatus

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(
        status="ok",
        version=settings.version,
        database="sqlite" if settings.database_url.startswith("sqlite") else "external",
        model=ModelStatus(**detector.status()),
    )


@router.get("/model", response_model=ModelStatus, summary="YOLO model status")
def model_status() -> ModelStatus:
    return ModelStatus(**detector.status())


@router.post("/model/reload", response_model=ModelStatus, summary="Reload YOLO weights")
def reload_model() -> ModelStatus:
    detector.load(force=True)
    return ModelStatus(**detector.status())


@router.get("/fruits/classes", response_model=list[ClassInfo], summary="Known classes")
def fruit_classes() -> list[ClassInfo]:
    detector.load()
    return [ClassInfo(**c) for c in detector.classes()]


@router.get("/fruits/known", response_model=list[str], summary="Canonical fruit names")
def known_fruits() -> list[str]:
    from ..detector import FRUIT_SYNONYMS

    return sorted(set(FRUIT_SYNONYMS.values()))
