# Roadmap

## Phase 0 — Scaffold (done)
YOLOv8m trainer, FastAPI `/predict` + `/health`, CLAHE preprocessing, Docker,
Gradio demo, pytest suite, CI. No trained weights yet — verified end-to-end
with a pretrained COCO model.

## Phase 1 — Data
Sources identified and download/merge tooling built — see
[DATA_SOURCES.md](DATA_SOURCES.md) and [prepare_data.py](prepare_data.py).
4 CC BY 4.0 Roboflow datasets covering all 3 classes, including a
DENTEX-Challenge-derived mirror for periapical_lesion. Run
`ROBOFLOW_API_KEY=... python prepare_data.py` to actually pull the data
(needs your own free Roboflow account key — not committed anywhere).
Merge any class under 100 instances into a neighbor rather than training on
it directly; the script prints per-class counts so you can check.

## Phase 2 — Baseline model
30-epoch baseline done (2026-09-14, `runs/detect/train9`, weights in
`models/best.pt`): mAP50 all=0.387 — calculus 0.704 (strong), caries 0.230
and periapical_lesion 0.228 (weak, recall 0.06-0.24). `train.py` currently
set to `epochs=30, batch=2, workers=0` (batch/workers forced down by an
8GB GPU + Windows spawn issues — see git history on `train.py` for the
full debugging trail if it acts up again).

**Next when resuming:** re-run longer (closer to the original 100 epochs)
now that the pipeline's proven sound, and look at why caries/periapical
recall is so low before trusting it — leading theory is the 65% background
image rate (images with only out-of-scope classes like crown/filling that
got stripped to empty labels) diluting those two classes specifically,
since calculus's source datasets didn't have that problem.

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
