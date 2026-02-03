from ultralytics import YOLO
import torch

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {'GPU' if device == 0 else 'CPU'}")

    # Load your model
    model = YOLO('C:/Users/jaysu/OneDrive/Desktop/yol/runs/segment/train2/weights/last.pt')

    # Resume training from the checkpoint
    model.train(
        resume=True,  # ✅ this resumes training automatically
        device=device
    )

if __name__ == "__main__":
    main()
