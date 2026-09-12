from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=1280,  # panoramic X-rays are wide
    batch=8,
)
