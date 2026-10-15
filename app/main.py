"""FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import init_db
from .detector import detector
from .routers import detect, meta, scans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("fruit_detection")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.app_name, settings.version)

    # 1. Prepare storage + database.
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    init_db()

    # 2. Warm the YOLO model in a background thread so the API accepts requests
    #    immediately instead of blocking for the (potentially slow) load.
    threading.Thread(target=_warm_model, name="yolo-warmup", daemon=True).start()

    yield
    logger.info("Shutting down.")


def _warm_model() -> None:
    try:
        detector.load()
        status = detector.status()
        if status["loaded"]:
            logger.info(
                "YOLO model '%s' loaded (%s classes).",
                status["model_name"],
                status["class_count"],
            )
        else:
            logger.warning("YOLO model not loaded: %s", status.get("error"))
    except Exception:  # pragma: no cover - defensive
        logger.exception("Unexpected error while warming the YOLO model")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Detect fruits in images with YOLO and keep a searchable history of "
        "every scan. Built for the Fruit Scanner Flutter app."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images (used by the app to show the scanned photo).
app.mount("/media", StaticFiles(directory=str(settings.upload_dir), check_dir=False), name="media")

app.include_router(meta.router)
app.include_router(detect.router)
app.include_router(scans.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
