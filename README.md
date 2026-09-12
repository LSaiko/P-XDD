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

## Data

```bash
pip install roboflow
ROBOFLOW_API_KEY=your_key python prepare_data.py
```

Pulls and merges the Phase 1 datasets into `dataset/` in the layout
`data.yaml` expects. See [DATA_SOURCES.md](DATA_SOURCES.md) for sources,
licenses, and citations.

## Training

```bash
python train.py
```

## Demo

```bash
python demo.py
```

## Layout

| File | Purpose |
|---|---|
| `main.py` | FastAPI inference server (CLAHE + YOLOv8) |
| `prepare_data.py`, `DATA_SOURCES.md` | Download and merge training data |
| `train.py`, `data.yaml` | Training |
| `demo.py` | Gradio UI |
| `test_main.py` | pytest suite (YOLO mocked) |
| `Dockerfile`, `docker-compose.yml` | Container build/run |
| `ROADMAP.md` | Phased plan and scale targets |
| `WORKFLOW.md` | Dev workflow |

## License

[MIT](LICENSE)
