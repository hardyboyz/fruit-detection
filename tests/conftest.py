"""Pytest fixtures.

Environment variables are set *before* the app is imported so the test run uses
a throwaway SQLite database and never writes real uploads.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="fruit-scanner-tests-"))

os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["MODEL_DIR"] = str(_TMP / "models")
os.environ["SAVE_UPLOADS"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import models  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    # Note: deliberately not used as a context manager, so the app lifespan
    # (which would warm the real YOLO model) does not run.
    init_db()
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_db(client: TestClient) -> None:
    """Start every test with empty tables."""
    db = SessionLocal()
    try:
        db.query(models.Detection).delete()
        db.query(models.Scan).delete()
        db.commit()
    finally:
        db.close()


class FakeDetector:
    """Replaces the real YOLO model in API tests."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image_bytes, conf=None, iou=None):
        from app.detector import Detection, PredictionResult

        self.calls += 1
        return PredictionResult(
            detections=[
                Detection(
                    label="Ripe Banana",
                    display_name="Banana",
                    category="fruit",
                    is_fruit=True,
                    confidence=0.9123,
                    class_id=2,
                    x1=10.0,
                    y1=20.0,
                    x2=110.0,
                    y2=120.0,
                ),
                Detection(
                    label="person",
                    display_name="Person",
                    category="other",
                    is_fruit=False,
                    confidence=0.41,
                    class_id=0,
                    x1=1.0,
                    y1=2.0,
                    x2=3.0,
                    y2=4.0,
                ),
            ],
            width=640,
            height=480,
            latency_ms=12.5,
            model_name="fake.pt",
        )


@pytest.fixture
def fake_detector(monkeypatch) -> FakeDetector:
    from app import detector as detector_module

    fake = FakeDetector()
    monkeypatch.setattr(detector_module.detector, "predict", fake.predict)
    return fake
