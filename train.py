from ultralytics import YOLO

if __name__ == "__main__":
    # guard required: DataLoader workers re-import this file with spawn (default on Windows)
    model = YOLO("yolov8m.pt")
    model.train(
        data="data.yaml",
        epochs=30,  # baseline run: fast first read before committing days to a full 100-epoch run
        imgsz=1280,  # panoramic X-rays are wide
        batch=2,  # 8 and 4 both OOM'd on this 8GB card at imgsz=1280; ultralytics' in-process
                  # auto-retry-on-OOM leaves the process fragmented rather than recovering cleanly,
                  # so set the batch size that fits directly instead of letting it cascade down
        workers=0,  # any >0 spawn workers exhausted the Windows paging file reloading torch+CUDA per worker
    )
