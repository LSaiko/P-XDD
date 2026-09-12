# Dental X-ray Pathology Detection

YOLOv8-based detector for **caries**, **calculus**, and **periapical lesions**
in dental panoramic X-rays, served over a FastAPI endpoint with CLAHE
preprocessing, plus a Gradio demo.

**Status:** scaffold only — no trained weights yet. `/predict` needs a real
`models/best.pt` to return meaningful detections. See [ROADMAP.md](ROADMAP.md).

## Quickstart

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

or with Docker:

```bash
docker compose up --build
```

## API

`POST /predict` — multipart file upload (`image/*`):

```json
{
  "detections": [
    {"class": "caries", "confidence": 0.87, "bbox": [12.0, 34.0, 88.0, 120.0]}
  ],
  "count": 1
}
```

`GET /health` — liveness check, independent of whether weights are loaded.

## Training

```bash
python train.py
```

Point `data.yaml` at your dataset (YOLO format — see [ROADMAP.md](ROADMAP.md)
Phase 1 for data sources).

## Demo

```bash
python demo.py
```

## Layout

| File | Purpose |
|---|---|
| `main.py` | FastAPI inference server (CLAHE + YOLOv8) |
| `train.py`, `data.yaml` | Training |
| `demo.py` | Gradio UI |
| `test_main.py` | pytest suite (YOLO mocked) |
| `Dockerfile`, `docker-compose.yml` | Container build/run |
| `ROADMAP.md` | Phased plan and scale targets |
| `WORKFLOW.md` | Dev workflow |

## License

[MIT](LICENSE)
