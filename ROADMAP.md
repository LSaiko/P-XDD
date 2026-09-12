# Roadmap

## Phase 0 — Scaffold (done)
YOLOv8m trainer, FastAPI `/predict` + `/health`, CLAHE preprocessing, Docker,
Gradio demo, pytest suite, CI. No trained weights yet — verified end-to-end
with a pretrained COCO model.

## Phase 1 — Data
Pull Roboflow "dental xray" datasets + the Dentex Challenge 2023 set. Target
1k+ images per class; merge any class under 100 instances into a neighbor
class rather than training on it directly.

## Phase 2 — Baseline model
Train `yolov8m` at imgsz 1280 / 100 epochs on the Phase 1 data. Hold out a
val split, record mAP50 per class as the baseline to beat.

## Phase 3 — API hardening
Only once a real model exists: auth/rate limiting on `/predict`, structured
logging, model versioning via `MODEL_PATH`, error metrics.

## Phase 4 — Demo & docs
Record the Gradio walkthrough, write a README with sample requests, publish
a model card (data sources, class definitions, known failure modes).

## Phase 5 — Scale
Only once there's real traffic: batch inference queue, GPU serving, model
monitoring, autoscaling. Don't build this early — it's pure waste against a
demo with no users.

## Depth-of-scale targets

| Tier | Data | Model | Serving | Move up when |
|---|---|---|---|---|
| Prototype (now) | few hundred imgs | yolov8n/m | uvicorn, single process | pipeline just needs to work |
| MVP | 1k+/class, augmented | yolov8m @1280 | Docker + gunicorn workers | need real accuracy for a demo/pitch |
| Production | multi-source, expert-reviewed | yolov8m/l, maybe ensembled | GPU serving, autoscaling, monitoring | real users, uptime/latency SLAs |

Stay at Prototype/MVP until something concrete demands the next tier.
