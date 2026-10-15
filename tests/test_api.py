"""End-to-end API tests using a fake detector (no YOLO weights required)."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient


def _image_upload(name: str = "photo.jpg"):
    # The detector is faked, so the bytes only need to pass content-type checks.
    return {"file": (name, io.BytesIO(b"\xff\xd8\xff\xe0fake-jpeg-bytes"), "image/jpeg")}


def _detect(client: TestClient, **data):
    fields = {"source": "camera", "device_id": "test-device", **data}
    return client.post("/api/detect", files=_image_upload(), data=fields)


def test_health(client: TestClient):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_detect_saves_scan(client: TestClient, fake_detector):
    res = _detect(client, note="unit test")
    assert res.status_code == 200, res.text

    body = res.json()
    assert body["model_name"] == "fake.pt"
    assert body["detection_count"] == 2
    assert body["fruit_count"] == 1
    assert body["image_width"] == 640
    assert body["latency_ms"] == 12.5

    # Detections come back sorted by confidence, with percentages precomputed.
    first = body["detections"][0]
    assert first["display_name"] == "Banana"
    assert first["confidence_percent"] == 91.2
    assert first["box"]["x2"] == 110.0
    assert body["created_at"].endswith("Z")  # UTC, so clients can localise

    assert fake_detector.calls == 1


def test_custom_confidence_threshold_is_forwarded(client: TestClient, monkeypatch):
    from app import detector as detector_module

    captured = {}

    def capture(image_bytes, conf=None, iou=None):
        captured["conf"] = conf
        from app.detector import PredictionResult

        return PredictionResult(width=10, height=10, latency_ms=1.0, model_name="fake.pt")

    monkeypatch.setattr(detector_module.detector, "predict", capture)
    res = client.post(
        "/api/detect",
        files=_image_upload(),
        data={"source": "camera", "conf": "0.7"},
    )
    assert res.status_code == 200
    assert captured["conf"] == 0.7


def test_preview_does_not_persist(client: TestClient, fake_detector):
    res = client.post("/api/detect/preview", files=_image_upload())
    assert res.status_code == 200
    assert len(res.json()["detections"]) == 2

    listing = client.get("/api/scans").json()
    assert listing["total"] == 0


def test_list_get_and_search(client: TestClient, fake_detector):
    _detect(client)
    _detect(client)

    listing = client.get("/api/scans").json()
    assert listing["total"] == 2
    assert listing["items"][0]["id"] > listing["items"][1]["id"]  # newest first

    scan_id = listing["items"][0]["id"]
    detail = client.get(f"/api/scans/{scan_id}").json()
    assert detail["id"] == scan_id
    assert [d["display_name"] for d in detail["detections"]] == ["Banana", "Person"]

    assert client.get("/api/scans", params={"search": "banana"}).json()["total"] == 2
    assert client.get("/api/scans", params={"search": "kiwi"}).json()["total"] == 0
    assert client.get("/api/scans", params={"fruits_only": True}).json()["total"] == 2


def test_stats(client: TestClient, fake_detector):
    _detect(client)
    _detect(client)

    stats = client.get("/api/scans/stats").json()
    assert stats["total_scans"] == 2
    assert stats["total_detections"] == 4
    assert stats["total_fruits"] == 2
    assert stats["top_fruits"][0]["name"] == "Banana"
    assert stats["top_fruits"][0]["count"] == 2


def test_delete_scan(client: TestClient, fake_detector):
    scan_id = _detect(client).json()["id"]

    assert client.delete(f"/api/scans/{scan_id}").status_code == 204
    assert client.get(f"/api/scans/{scan_id}").status_code == 404
    assert client.get("/api/scans").json()["total"] == 0


def test_delete_all_for_device(client: TestClient, fake_detector):
    _detect(client)
    _detect(client)
    res = client.delete("/api/scans", params={"device_id": "test-device"})
    assert res.status_code == 200
    assert res.json()["deleted"] == 2
    assert client.get("/api/scans").json()["total"] == 0


def test_rejects_non_image(client: TestClient):
    res = client.post(
        "/api/detect",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert res.status_code == 415


def test_rejects_empty_file(client: TestClient):
    res = client.post(
        "/api/detect",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert res.status_code == 400


def test_missing_scan_returns_404(client: TestClient):
    assert client.get("/api/scans/99999").status_code == 404
