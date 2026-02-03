from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os

def main():
    # ✅ Paths
    model_path = r"C:/Users/jaysu/OneDrive/Desktop/yol/runs/segment/train/weights/best.pt"
    image_path = r"C:/Users/jaysu/OneDrive/Desktop/yol/testing/6.jpg"
    output_path = r"C:/Users/jaysu/OneDrive/Desktop/yol/output/lane_overlay.png"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ✅ Load image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Could not read image. Check your image_path.")
        return

    H, W, _ = img.shape

    # ✅ Load model (use GPU if available)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {'GPU' if device == 0 else 'CPU'}")

    model = YOLO(model_path)

    # ✅ Run inference
    results = model(img, device=device)

    for result in results:
        if result.masks is None:
            print("⚠️ No lane segmentation detected.")
            continue

        # ✅ Create a copy of the original image
        overlay = img.copy()

        for mask in result.masks.data:
            mask = (mask.cpu().numpy() * 255).astype("uint8")
            mask = cv2.resize(mask, (W, H))

            # ✅ Create colored overlay (blue lanes)
            color_mask = np.zeros_like(img)
            color_mask[:, :, 2] = mask  # Red channel (change to 0/1/2 for blue/green/red)

            # ✅ Blend mask with original image
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)

        # ✅ Show the overlaid image
        cv2.imshow("Lane Detection Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # ✅ Save result
        cv2.imwrite(output_path, overlay)
        print(f"✅ Saved overlaid image to: {output_path}")

if __name__ == "__main__":
    main()
