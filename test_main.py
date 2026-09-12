from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _jpeg_bytes():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _fake_box(cls_id, conf, bbox):
    box = MagicMock()
    box.cls = np.array([cls_id])
    box.conf = np.array([conf])
    box.xyxy = [np.array(bbox, dtype=float)]
    return box


def _fake_model():
    model = MagicMock()
    model.names = {0: "caries"}
    result = MagicMock(boxes=[_fake_box(0, 0.9, [1.0, 2.0, 3.0, 4.0])])
    model.return_value = [result]
    return model


def test_predict_valid_jpeg_returns_detections():
    with patch("main.get_model", return_value=_fake_model()):
        resp = client.post(
            "/predict", files={"file": ("x.jpg", _jpeg_bytes(), "image/jpeg")}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["detections"][0]["class"] == "caries"


def test_predict_empty_file_returns_400():
    resp = client.post("/predict", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert resp.status_code == 400


def test_predict_non_image_returns_422():
    resp = client.post(
        "/predict", files={"file": ("x.txt", b"not an image", "text/plain")}
    )
    assert resp.status_code == 422


def test_predict_response_schema():
    with patch("main.get_model", return_value=_fake_model()):
        resp = client.post(
            "/predict", files={"file": ("x.jpg", _jpeg_bytes(), "image/jpeg")}
        )
    body = resp.json()
    det = body["detections"][0]
    assert set(det.keys()) == {"class", "confidence", "bbox"}
    assert isinstance(det["bbox"], list) and len(det["bbox"]) == 4
