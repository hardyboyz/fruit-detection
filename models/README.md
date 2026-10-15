# YOLO weights live here.
#
# * `fruit_yolo.pt`  — the dedicated fruit model (primary). It knows banana,
#   mango and pomegranate (plus ripeness). Any file matching `*fruit*.pt` is
#   picked up automatically; put your own trained detector here and restart.
# * `yolo11s.pt`     — a general COCO model used as the *secondary* detector so
#   fruit the custom model never learned (apple, orange, ...) are still found.
#   It is downloaded automatically on first run if missing.
#
# No custom model? That's fine: the API uses `yolo11s.pt` alone, which knows
# apple / banana / orange. See ../README.md for how the ensemble works.
