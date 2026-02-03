from ultralytics import YOLO
import torch

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {'GPU' if device == 0 else 'CPU'}")


    model = YOLO('C:/Users/jaysu/OneDrive/Desktop/yol/runs/segment/train2/weights/last.pt')

    model.train(
        resume=True,
        device=device
    )

if __name__ == "__main__":
    main()
