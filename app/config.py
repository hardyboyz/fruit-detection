"""Application configuration.

All values can be overridden with environment variables so the same code can run
on a laptop, in Docker, or on a server without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/  (the parent folder of the `app` package)
BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    """Runtime settings resolved from the environment."""

    app_name: str = "Fruit Detection API"
    version: str = "1.0.0"

    base_dir: Path = BASE_DIR

    # --- Storage -----------------------------------------------------------
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'fruits.db').as_posix()}"
    )
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "storage" / "uploads")))

    # --- YOLO --------------------------------------------------------------
    # Where to look for a *dedicated fruit* YOLO model.
    model_dir: Path = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models")))
    # Explicit path to a custom fruit model (takes priority when it exists).
    yolo_weights: str = os.getenv("YOLO_WEIGHTS", "").strip()
    # Used when no custom weights are found. `yolov8n.pt` is auto-downloaded by
    # ultralytics on first run and knows a handful of fruit classes (COCO).
    fallback_weights: str = os.getenv("YOLO_FALLBACK_WEIGHTS", "yolov8n.pt")
    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.25"))
    iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    imgsz: int = int(os.getenv("IMGSZ", "640"))
    device: str = os.getenv("YOLO_DEVICE", "").strip()  # "" = auto, "cpu", "mps", "0"

    # --- Ensemble (multi-model) -------------------------------------------
    # A dedicated fruit model knows its own classes well but is *forced* to
    # guess when it sees a fruit it was never trained on (e.g. an apple became
    # "pomegranate"). We fix that by running a general COCO detector next to it
    # and averaging a few ImageNet classifiers over each box to arbitrate.
    ensemble_enabled: bool = _env_bool("ENSEMBLE_ENABLED", True)
    # General detector (COCO) paired with the custom fruit model.
    secondary_weights: str = os.getenv("YOLO_SECONDARY_WEIGHTS", "yolo11s.pt")
    # Where to fetch it from when it is not already inside `models/`.
    secondary_url: str = os.getenv(
        "YOLO_SECONDARY_URL",
        "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s.pt",
    )
    # torchvision ImageNet classifiers whose averaged probabilities pick the
    # final class. Any subset of torchvision model names works.
    classifier_enabled: bool = _env_bool("CLASSIFIER_ENABLED", True)
    classifier_models: list[str] = _env_list(
        "CLASSIFIER_MODELS",
        ["efficientnet_b0", "efficientnet_b4", "convnext_tiny"],
    )
    # Confidence the custom model needs before we trust a "mango". ImageNet has
    # no mango class, so mango is the one fruit only the custom model can name.
    mango_trust_conf: float = float(os.getenv("MANGO_TRUST_CONF", "0.90"))
    # IoU at/above which two detectors' boxes describe the same object.
    cluster_iou: float = float(os.getenv("CLUSTER_IOU", "0.60"))

    # --- Uploads -----------------------------------------------------------
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "12"))
    save_uploads: bool = _env_bool("SAVE_UPLOADS", True)

    # --- CORS --------------------------------------------------------------
    # "*" keeps mobile/dev clients happy; tighten this in production.
    cors_origins: list[str] = _env_list("CORS_ORIGINS", ["*"])

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
