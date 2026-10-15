"""Fruit detector.

Two layers work together:

1. **Localisation** — YOLO models find *where* the fruit is.
   * A dedicated fruit model (``YOLO_WEIGHTS``, or ``*fruit*.pt`` dropped in
     ``models/``) is the primary detector.
   * When that model is present we also run a general COCO detector
     (``yolo11s.pt``) next to it, because a small dedicated model is often
     *forced* to mislabel fruit it was never trained on (a red apple became
     "pomegranate" because the model has no apple class).

2. **Arbitration** — a small ImageNet classifier ensemble (torchvision) looks
   at each detected box and votes on the class. Its opinion breaks the tie
   between the two YOLO models, so classes neither model knows reliably
   (apple, orange, ...) are still named correctly.

Everything degrades gracefully: if torchvision or the secondary weights are
unavailable, the detector simply behaves like the single-model version.

The detector is process-wide and lazily loaded so importing the app stays fast.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label normalisation (English model labels -> Indonesian display names)
# ---------------------------------------------------------------------------

# Canonical *English* fruit names. Keys are matched case-insensitively as
# substrings so "red_apple", "Granny Smith apple" and "Apple " all collapse to
# "Apple". Order matters: longer / more specific keys are checked first.
FRUIT_SYNONYMS: dict[str, str] = {
    "green apple": "Apple",
    "red apple": "Apple",
    "granny smith": "Apple",
    "golden delicious": "Apple",
    "apple": "Apple",
    "banana": "Banana",
    "plantain": "Banana",
    "orange": "Orange",
    "tangerine": "Orange",
    "mandarin": "Orange",
    "clementine": "Orange",
    "lemon": "Lemon",
    "lime": "Lime",
    "grapefruit": "Grapefruit",
    "pomelo": "Pomelo",
    "mango": "Mango",
    "papaya": "Papaya",
    "pineapple": "Pineapple",
    "watermelon": "Watermelon",
    "melon": "Melon",
    "cantaloupe": "Melon",
    "honeydew": "Melon",
    "strawberry": "Strawberry",
    "blueberry": "Blueberry",
    "blackberry": "Blackberry",
    "raspberry": "Raspberry",
    "cherry": "Cherry",
    "grape": "Grape",
    "kiwi": "Kiwi",
    "peach": "Peach",
    "nectarine": "Nectarine",
    "pear": "Pear",
    "plum": "Plum",
    "apricot": "Apricot",
    "pomegranate": "Pomegranate",
    "fig": "Fig",
    "date": "Date",
    "avocado": "Avocado",
    "coconut": "Coconut",
    "dragon fruit": "Dragon Fruit",
    "dragonfruit": "Dragon Fruit",
    "lychee": "Lychee",
    "guava": "Guava",
    "passion fruit": "Passion Fruit",
    "persimmon": "Persimmon",
    "cranberry": "Cranberry",
    "durian": "Durian",
    "rambutan": "Rambutan",
    "jackfruit": "Jackfruit",
    "starfruit": "Starfruit",
    "carambola": "Starfruit",
    "custard apple": "Custard Apple",
    "sugar apple": "Custard Apple",
}

# Common non-fruit produce we can still recognise / normalise.
VEGETABLE_SYNONYMS: dict[str, str] = {
    "broccoli": "Broccoli",
    "carrot": "Carrot",
    "tomato": "Tomato",
    "cucumber": "Cucumber",
    "bell pepper": "Bell Pepper",
    "pepper": "Bell Pepper",
    "potato": "Potato",
    "onion": "Onion",
    "garlic": "Garlic",
    "pumpkin": "Pumpkin",
    "eggplant": "Eggplant",
    "aubergine": "Eggplant",
    "cabbage": "Cabbage",
    "lettuce": "Lettuce",
    "corn": "Corn",
    "mushroom": "Mushroom",
    "zucchini": "Zucchini",
    "ginger": "Ginger",
}

# Canonical English name -> Indonesian display name.
INDONESIAN_NAMES: dict[str, str] = {
    "Apple": "Apel",
    "Banana": "Pisang",
    "Orange": "Jeruk",
    "Lemon": "Lemon",
    "Lime": "Jeruk Nipis",
    "Grapefruit": "Jeruk Gedang",
    "Pomelo": "Jeruk Bali",
    "Mango": "Mangga",
    "Papaya": "Pepaya",
    "Pineapple": "Nanas",
    "Watermelon": "Semangka",
    "Melon": "Melon",
    "Strawberry": "Stroberi",
    "Blueberry": "Blueberry",
    "Blackberry": "Blackberry",
    "Raspberry": "Raspberry",
    "Cherry": "Ceri",
    "Grape": "Anggur",
    "Kiwi": "Kiwi",
    "Peach": "Persik",
    "Nectarine": "Nektarin",
    "Pear": "Pir",
    "Plum": "Plum",
    "Apricot": "Aprikot",
    "Pomegranate": "Delima",
    "Fig": "Buah Tin",
    "Date": "Kurma",
    "Avocado": "Alpukat",
    "Coconut": "Kelapa",
    "Dragon Fruit": "Buah Naga",
    "Lychee": "Leci",
    "Guava": "Jambu Biji",
    "Passion Fruit": "Markisa",
    "Persimmon": "Kesemek",
    "Cranberry": "Cranberry",
    "Durian": "Durian",
    "Rambutan": "Rambutan",
    "Jackfruit": "Nangka",
    "Starfruit": "Belimbing",
    "Custard Apple": "Srikaya",
}

INDONESIAN_VEGETABLES: dict[str, str] = {
    "Broccoli": "Brokoli",
    "Carrot": "Wortel",
    "Tomato": "Tomat",
    "Cucumber": "Timun",
    "Bell Pepper": "Paprika",
    "Potato": "Kentang",
    "Onion": "Bawang Bombay",
    "Garlic": "Bawang Putih",
    "Pumpkin": "Labu",
    "Eggplant": "Terong",
    "Cabbage": "Kubis",
    "Lettuce": "Selada",
    "Corn": "Jagung",
    "Mushroom": "Jamur",
    "Zucchini": "Zucchini",
    "Ginger": "Jahe",
}

# Labels a general COCO detector emits that are definitely *not* produce. They
# must not be counted as fruit just because we don't recognise them.
NON_PRODUCE: frozenset[str] = frozenset(
    {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    }
)

# Ripeness qualifiers emitted by fruit models (e.g. "Ripe Banana"). Checked in
# order, first match wins. Keys are matched against the normalised token, so
# hyphens/underscores are already spaces by the time we get here.
RIPENESS_TOKENS: list[tuple[str, str]] = [
    ("unripe", "Mentah"),
    ("raw", "Mentah"),
    ("early fruit", "Muda"),
    ("mid growth", "Setengah Matang"),
    ("ripe", "Matang"),
]

# ImageNet classes (torchvision) we accept as fruits, so that unrelated ImageNet
# labels ("spaghetti squash", "hot dog", ...) are ignored by the arbiter.
IMAGENET_FRUIT_NAMES: frozenset[str] = frozenset(
    {
        "granny smith",
        "orange",
        "lemon",
        "banana",
        "pomegranate",
        "fig",
        "pineapple",
        "strawberry",
        "jackfruit",
        "custard apple",
    }
)


def _normalise_token(label: str) -> str:
    return label.strip().lower().replace("_", " ").replace("-", " ")


def _match(token: str, table: dict[str, str]) -> str | None:
    for key, canonical in sorted(table.items(), key=lambda kv: -len(kv[0])):
        if key in token:
            return canonical
    return None


def _ripeness(token: str) -> str:
    # Only match at the start of the label ("Ripe Banana"), never mid-word
    # ("strawberry" must not look "raw").
    for key, value in RIPENESS_TOKENS:
        if token == key or token.startswith(key + " "):
            return value
    return ""


def canonical_fruit(label: str) -> str | None:
    """Return the canonical *English* fruit name for a raw label, if known."""
    return _match(_normalise_token(label), FRUIT_SYNONYMS)


def is_produce(label: str) -> bool:
    """True when a raw label maps to a known fruit or vegetable."""
    token = _normalise_token(label)
    return (
        _match(token, FRUIT_SYNONYMS) is not None
        or _match(token, VEGETABLE_SYNONYMS) is not None
    )


def classify_label(label: str) -> tuple[str, str, bool]:
    """Return ``(display_name, category, is_fruit)`` for a raw model label.

    The display name is the Indonesian fruit/vegetable name, optionally with a
    ripeness qualifier (e.g. ``"Ripe Banana"`` -> ``"Pisang Matang"``).
    """
    token = _normalise_token(label)

    canonical = _match(token, FRUIT_SYNONYMS)
    if canonical:
        base = INDONESIAN_NAMES.get(canonical, canonical)
        ripe = _ripeness(token)
        return (f"{base} {ripe}" if ripe else base), "fruit", True

    canonical = _match(token, VEGETABLE_SYNONYMS)
    if canonical:
        base = INDONESIAN_VEGETABLES.get(canonical, canonical)
        ripe = _ripeness(token)
        return (f"{base} {ripe}" if ripe else base), "vegetable", False

    if token in NON_PRODUCE:
        return (token.title() if token else label), "other", False

    # Unknown label (custom model with a class we don't know about). Title-case
    # it and treat it as a fruit by default so nothing is silently dropped.
    pretty = token.title() if token else label
    return pretty, "fruit", True


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    label: str
    display_name: str
    category: str
    is_fruit: bool
    confidence: float
    class_id: int | None
    x1: float
    y1: float
    x2: float
    y2: float

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "display_name": self.display_name,
            "category": self.category,
            "is_fruit": self.is_fruit,
            "confidence": round(self.confidence, 4),
            "confidence_percent": round(self.confidence * 100, 1),
            "class_id": self.class_id,
            "box": {
                "x1": round(self.x1, 2),
                "y1": round(self.y1, 2),
                "x2": round(self.x2, 2),
                "y2": round(self.y2, 2),
            },
        }


@dataclass
class PredictionResult:
    detections: list[Detection] = field(default_factory=list)
    width: int = 0
    height: int = 0
    latency_ms: float = 0.0
    model_name: str = ""


@dataclass
class _Claim:
    """One model's opinion about one box."""

    box: tuple[float, float, float, float]
    label: str
    confidence: float
    class_id: int | None
    source: str  # "primary" | "secondary"


class DetectorUnavailable(RuntimeError):
    """Raised when no usable YOLO model could be loaded."""


class FruitDetector:
    """Thread-safe, lazily-loaded multi-model fruit detector."""

    def __init__(self) -> None:
        self._model = None
        self._names: dict[int, str] = {}
        self._secondary = None
        self._secondary_names: dict[int, str] = {}
        self._classifier = None  # list[(model, transform, categories)]
        self._lock = threading.Lock()
        self._load_attempted = False

        self.model_name: str | None = None
        self.weights_path: str | None = None
        self.is_custom_fruit_model: bool = False
        self.secondary_model_name: str | None = None
        self.secondary_weights_path: str | None = None
        self.classifier_names: list[str] = []
        self.device: str | None = None
        self.error: str | None = None

    # -- model discovery ---------------------------------------------------

    def _reserved_names(self) -> set[str]:
        return {
            Path(settings.secondary_weights).name.lower(),
            Path(settings.fallback_weights).name.lower(),
        }

    def _discover_custom_weights(self) -> Path | None:
        """Find a dedicated fruit model, if one has been provided."""
        if settings.yolo_weights:
            candidate = Path(settings.yolo_weights).expanduser()
            if candidate.exists():
                return candidate
            logger.warning("YOLO_WEIGHTS points to a missing file: %s", candidate)

        model_dir = Path(settings.model_dir)
        reserved = self._reserved_names()
        if model_dir.is_dir():
            # Prefer a file that clearly names itself as a fruit model, then any
            # other checkpoint the user dropped in (excluding the general
            # detector, which is discovered separately).
            for pattern in (
                "*fruit*.pt",
                "*fruits*.pt",
                "*fruit*.onnx",
                "*fruits*.onnx",
                "*.pt",
                "*.onnx",
            ):
                for match in sorted(model_dir.glob(pattern)):
                    if match.name.lower() in reserved:
                        continue
                    return match
        return None

    def _resolve_fallback(self) -> str:
        """Resolve the general (COCO) weights used as the primary detector when
        no custom fruit model exists, or as the secondary detector when one does.
        """
        fallback = settings.fallback_weights
        explicit = Path(fallback).expanduser()
        if explicit.exists():
            return str(explicit)

        candidate = Path(settings.model_dir) / explicit.name
        if candidate.exists():
            return str(candidate)

        # Reuse the already-downloaded general detector instead of relying on
        # ultralytics' auto-download (which uses a host that is often blocked).
        secondary = Path(settings.model_dir) / Path(settings.secondary_weights).name
        if secondary.exists():
            return str(secondary)

        return fallback

    def _resolve_secondary(self) -> str | None:
        """Locate (or download) the general COCO detector."""
        name = settings.secondary_weights
        explicit = Path(name).expanduser()
        if explicit.exists():
            return str(explicit)

        candidate = Path(settings.model_dir) / explicit.name
        if candidate.exists():
            return str(candidate)

        if not settings.secondary_url:
            return None

        dest = Path(settings.model_dir) / Path(name).name
        try:
            import requests

            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading general detector: %s", settings.secondary_url)
            with requests.get(settings.secondary_url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            return str(dest)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Could not obtain the general detector: %s", exc)
            return None

    def _load_classifier(self) -> list | None:
        """Load the torchvision ImageNet classifier ensemble (optional)."""
        try:
            import torchvision.models as tvm
            from torchvision.models import get_model_weights
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.warning("torchvision is not available (%s); skipping arbiter.", exc)
            return None

        loaded: list = []
        for name in settings.classifier_models:
            factory = getattr(tvm, name, None)
            if factory is None:
                logger.warning("Unknown classifier model: %s", name)
                continue
            try:
                weights = get_model_weights(factory).DEFAULT
                model = factory(weights=weights).eval()
                loaded.append((model, weights.transforms(), list(weights.meta["categories"])))
                logger.info("Classifier ready: %s", name)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Could not load classifier %s: %s", name, exc)
        return loaded or None

    def load(self, force: bool = False) -> None:
        """Load the models once. Safe to call from multiple threads."""
        with self._lock:
            if self._load_attempted and not force:
                return
            self._load_attempted = True
            self.error = None

            try:
                from ultralytics import YOLO  # imported lazily: heavy dependency
            except Exception as exc:  # pragma: no cover - depends on environment
                self.error = (
                    "ultralytics/torch is not installed. Run "
                    "`pip install -r requirements.txt`."
                )
                logger.error("YOLO import failed: %s", exc)
                return

            custom = self._discover_custom_weights()
            weights = str(custom) if custom else self._resolve_fallback()

            try:
                logger.info("Loading YOLO weights: %s", weights)
                model = YOLO(weights)
                if settings.device:
                    model.to(settings.device)
                self._model = model
                self._names = {int(k): str(v) for k, v in dict(model.names).items()}
                self.weights_path = weights
                self.is_custom_fruit_model = custom is not None
                self.model_name = Path(weights).name
                self.device = settings.device or "auto"
                logger.info(
                    "Primary model ready (%s classes, custom=%s)",
                    len(self._names),
                    self.is_custom_fruit_model,
                )
            except Exception as exc:  # pragma: no cover - depends on environment
                self._model = None
                self.error = f"Failed to load YOLO weights '{weights}': {exc}"
                logger.exception("Failed to load YOLO model")
                return

            # A dedicated fruit model is the only case that benefits from the
            # ensemble: it is the one that gets *forced* into wrong classes.
            if settings.ensemble_enabled and self.is_custom_fruit_model:
                secondary = self._resolve_secondary()
                if secondary:
                    try:
                        sec = YOLO(secondary)
                        if settings.device:
                            sec.to(settings.device)
                        self._secondary = sec
                        self._secondary_names = {
                            int(k): str(v) for k, v in dict(sec.names).items()
                        }
                        self.secondary_weights_path = secondary
                        self.secondary_model_name = Path(secondary).name
                        logger.info("General detector ready: %s", self.secondary_model_name)
                    except Exception as exc:  # pragma: no cover
                        logger.warning("Could not load the general detector: %s", exc)

            if (
                settings.ensemble_enabled
                and settings.classifier_enabled
                and self._secondary is not None
            ):
                self._classifier = self._load_classifier()
                if self._classifier:
                    self.classifier_names = [Path(w).stem for w in settings.classifier_models]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- inference helpers -------------------------------------------------

    def _collect_claims(
        self,
        model,
        names: dict[int, str],
        source: str,
        image: Image.Image,
        conf: float,
        iou: float,
        out: list[_Claim],
        produce_only: bool = False,
    ) -> None:
        results = model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=settings.imgsz,
            verbose=False,
        )
        if not results:
            return
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, classes):
            label = names.get(int(cls_id), str(cls_id))
            # A general detector also knows people, cars, ... — ignore those so
            # the scan stays about produce.
            if produce_only and not is_produce(label):
                continue
            out.append(
                _Claim(
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    label=label,
                    confidence=float(score),
                    class_id=int(cls_id),
                    source=source,
                )
            )

    @staticmethod
    def _cluster(claims: list[_Claim], iou_threshold: float) -> list[list[_Claim]]:
        """Group overlapping claims (from either model) into objects."""
        clusters: list[list[_Claim]] = []
        for claim in sorted(claims, key=lambda c: c.confidence, reverse=True):
            for cluster in clusters:
                if _iou(claim.box, cluster[0].box) >= iou_threshold:
                    cluster.append(claim)
                    break
            else:
                clusters.append([claim])
        return clusters

    @staticmethod
    def _crop(image: Image.Image, box, pad: float = 0.08) -> Image.Image:
        width, height = image.size
        x1, y1, x2, y2 = box
        dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
        return image.crop(
            (
                max(0, int(x1 - dx)),
                max(0, int(y1 - dy)),
                min(width, int(x2 + dx)),
                min(height, int(y2 + dy)),
            )
        )

    def _classify(self, crop: Image.Image) -> tuple[str, float] | None:
        """Return ``(imagenet_label, probability)`` for a fruit, else ``None``."""
        if not self._classifier:
            return None
        import torch

        probs = None
        for model, transform, _categories in self._classifier:
            tensor = transform(crop.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                step = torch.softmax(model(tensor)[0], 0)
            probs = step if probs is None else probs + step
        # All torchvision ImageNet weights share the same 1000-class order.
        categories = self._classifier[0][2]
        index = int(torch.argmax(probs))
        name = categories[index]
        if _normalise_token(name) in IMAGENET_FRUIT_NAMES:
            return name, float(probs[index]) / len(self._classifier)
        return None

    # -- public API --------------------------------------------------------

    def predict(
        self,
        image_bytes: bytes,
        conf: float | None = None,
        iou: float | None = None,
    ) -> PredictionResult:
        self.load()
        if self._model is None:
            raise DetectorUnavailable(self.error or "YOLO model is not available.")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Could not read image: {exc}") from exc
        width, height = image.size

        conf = settings.conf_threshold if conf is None else conf
        iou = settings.iou_threshold if iou is None else iou

        started = time.perf_counter()

        claims: list[_Claim] = []
        self._collect_claims(self._model, self._names, "primary", image, conf, iou, claims)
        if self._secondary is not None:
            self._collect_claims(
                self._secondary,
                self._secondary_names,
                "secondary",
                image,
                conf,
                iou,
                claims,
                produce_only=True,
            )

        detections: list[Detection] = []
        for cluster in self._cluster(claims, settings.cluster_iou):
            detection = self._resolve(cluster, image)
            if detection is not None:
                detections.append(detection)

        # Nothing localised? Fall back to classifying the whole image.
        if not detections and self._classifier:
            classified = self._classify(image)
            if classified is not None:
                raw, prob = classified
                display, category, is_fruit = classify_label(raw)
                detections.append(
                    Detection(
                        label=raw,
                        display_name=display,
                        category=category,
                        is_fruit=is_fruit,
                        confidence=prob,
                        class_id=None,
                        x1=0.0,
                        y1=0.0,
                        x2=float(width),
                        y2=float(height),
                    )
                )

        latency_ms = (time.perf_counter() - started) * 1000.0
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return PredictionResult(
            detections=detections,
            width=width,
            height=height,
            latency_ms=round(latency_ms, 2),
            model_name=self.model_name or "",
        )

    def _resolve(self, cluster: list[_Claim], image: Image.Image) -> Detection | None:
        """Turn one object's overlapping claims into a single detection."""
        primary = max(
            (c for c in cluster if c.source == "primary"),
            key=lambda c: c.confidence,
            default=None,
        )
        secondary = max(
            (c for c in cluster if c.source == "secondary"),
            key=lambda c: c.confidence,
            default=None,
        )
        best = max(cluster, key=lambda c: c.confidence)
        score = best.confidence
        box = best.box

        primary_fruit = canonical_fruit(primary.label) if primary else None
        secondary_fruit = canonical_fruit(secondary.label) if secondary else None
        conflict = (
            primary is not None
            and secondary is not None
            and primary_fruit != secondary_fruit
        )

        # Only pay for the (slow) classifier when it can change the answer:
        # when the two detectors disagree, or when the custom model claims one
        # of the classes it is known to over-fire (it has no apple/orange, so
        # those become mango/pomegranate).
        needs_arbiter = self._classifier is not None and (
            conflict or primary_fruit in {"Mango", "Pomegranate"}
        )
        classified = self._classify(self._crop(image, box)) if needs_arbiter else None
        class_label = classified[0] if classified else None
        class_fruit = canonical_fruit(class_label) if class_label else None

        chosen: _Claim | None
        if (
            primary is not None
            and primary_fruit == "Mango"
            and primary.confidence >= settings.mango_trust_conf
            and class_fruit not in {"Apple", "Pomegranate"}
        ):
            # ImageNet has no mango class, so a confident mango from the custom
            # model wins unless the arbiter strongly says apple/pomegranate.
            chosen = primary
            raw_label = primary.label
        elif (
            primary is not None
            and primary_fruit is not None
            and not conflict
            and primary_fruit not in {"Mango", "Pomegranate"}
        ):
            # Both models agree (or only the fruit model fired): keep its label
            # so ripeness ("Ripe Banana") is preserved.
            chosen = primary
            raw_label = primary.label
        elif class_fruit is not None and class_fruit == primary_fruit and primary is not None:
            chosen = primary
            raw_label = primary.label
        elif class_label is not None:
            chosen = primary or secondary
            raw_label = class_label
        elif secondary is not None:
            chosen = secondary
            raw_label = secondary.label
        elif primary is not None:
            chosen = primary
            raw_label = primary.label
        else:  # pragma: no cover - a cluster always has at least one claim
            return None

        display, category, is_fruit = classify_label(raw_label)
        return Detection(
            label=raw_label,
            display_name=display,
            category=category,
            is_fruit=is_fruit,
            confidence=score,
            class_id=chosen.class_id if chosen else None,
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
        )

    def status(self) -> dict:
        fruit_class_count = sum(
            1 for name in self._names.values() if classify_label(name)[2]
        )
        return {
            "loaded": self.is_loaded,
            "model_name": self.model_name,
            "weights_path": self.weights_path,
            "is_custom_fruit_model": self.is_custom_fruit_model,
            "class_count": len(self._names),
            "fruit_class_count": fruit_class_count,
            "device": self.device,
            "error": self.error,
            "ensemble_active": self._secondary is not None,
            "secondary_model_name": self.secondary_model_name,
            "secondary_weights_path": self.secondary_weights_path,
            "classifier_models": self.classifier_names,
        }

    def classes(self) -> list[dict]:
        out: list[dict] = []
        seen: set[str] = set()
        for cid in sorted(self._names):
            raw = self._names[cid]
            display, category, is_fruit = classify_label(raw)
            if display in seen:
                continue
            seen.add(display)
            out.append({"name": display, "category": category, "is_fruit": is_fruit})
        return out


# Module-level singleton used by the routers.
detector = FruitDetector()
