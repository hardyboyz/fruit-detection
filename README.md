# Backend — Fruit Detection API

FastAPI + Ultralytics YOLO + SQLite. See the [project README](../README.md) for
the full walkthrough.

> Requires **Python 3.10+**. Do **not** use the system `python3` (3.9 on macOS).

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # or call .venv/bin/python directly
pip install -r requirements.txt
python run.py                      # http://127.0.0.1:8000/docs
```

## Layout

| Path | Purpose |
| --- | --- |
| `app/main.py` | App, CORS, `/media` static mount, startup |
| `app/config.py` | Environment-driven settings |
| `app/detector.py` | Multi-model YOLO wrapper + Indonesian fruit label normalisation |
| `app/models.py` | `Scan` / `Detection` ORM models |
| `app/routers/` | `detect.py`, `scans.py`, `meta.py` |
| `models/` | **Drop your YOLO weights here** (`fruit_yolo.pt`, `yolo11s.pt`) |
| `storage/uploads/` | Scan images |
| `tests/` | Pytest suite (uses a fake detector — no weights needed) |

## How detection works

A dedicated fruit model is great at its own classes but is *forced* to
mislabel fruit it was never trained on — the bundled 7-class
`fruit_yolo.pt` has no `apple`, so a red apple became **"pomegranate"**. The
detector therefore runs a small ensemble:

1. **Primary** — your fruit model (`fruit_yolo.pt`) finds boxes and ripeness.
2. **Secondary** — a general COCO model (`yolo11s.pt`) runs alongside it and
   finds fruit the custom model cannot represent (apple, orange, ...).
3. **Arbiter** — a torchvision ImageNet classifier ensemble looks at each box
   and votes. It is only invoked when the two models disagree (or the custom
   model claims its common false positives), so ordinary scans stay fast.

Everything degrades gracefully: if `yolo11s.pt`/torchvision are missing the
detector behaves like the single-model version. Display names are Indonesian
(`Apple` → `Apel`, `Ripe Banana` → `Pisang Matang`, ...). Tune it via
`ENSEMBLE_ENABLED`, `CLASSIFIER_ENABLED`, `MANGO_TRUST_CONF`, etc. in
`app/config.py`. Models are fetched on first run from Hugging Face /
`download.pytorch.org` and then cached.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Endpoints

`GET /api/health` · `GET /api/model` · `POST /api/model/reload` ·
`GET /api/fruits/classes` · `POST /api/detect` · `POST /api/detect/preview` ·
`GET /api/scans` · `GET /api/scans/stats` · `GET /api/scans/{id}` ·
`DELETE /api/scans/{id}` · `DELETE /api/scans`

Full schema: <http://127.0.0.1:8000/docs>
