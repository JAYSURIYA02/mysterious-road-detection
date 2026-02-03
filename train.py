from ultralytics import YOLO
import torch

def main():

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {'GPU' if device == 0 else 'CPU'}")

    model = YOLO('yolov8s-seg.pt')

    model.train(
        data='config.yaml',  # full path to config
        epochs=75,
        imgsz=640,
        patience=30,
        device=0
    )

if __name__ == "__main__":
    main()
