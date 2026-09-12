import os

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from ultralytics import YOLO

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")

app = FastAPI()
_model = None


def get_model():
    # ponytail: lazy-loaded so /health works even before weights exist (e.g. fresh container)
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def clahe_preprocess(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    class_: str = Field(alias="class")
    confidence: float
    bbox: list[float]


class PredictionResponse(BaseModel):
    detections: list[Detection]
    count: int


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(422, "file must be an image")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "empty file")

    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "invalid image")

    img = clahe_preprocess(img)
    model = get_model()
    results = model(img)[0]
    detections = [
        {
            "class": model.names[int(b.cls[0])],
            "confidence": float(b.conf[0]),
            "bbox": b.xyxy[0].tolist(),
        }
        for b in results.boxes
    ]
    return {"detections": detections, "count": len(detections)}
