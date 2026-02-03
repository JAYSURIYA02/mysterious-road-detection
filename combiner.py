import torch
import cv2
import time
from ultralytics import YOLO
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device}")


# change to lane segmentation model path
lane_seg_model_path = r"runs\segment\train2\weights\best.pt"
model_lane = YOLO(lane_seg_model_path).to(device)
print("YOLOv8 lane segmentation model loaded.")

# change to pothole/speed bump detection model path
detect_model_path = r"runs\detect\train3\weights\best.pt"
model_detect = YOLO(detect_model_path).to(device)
print("YOLOv8 pothole/speed bump model loaded.")

# Change video file path as needed
video_path = r"testing\8.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Could not open video.")
    exit()

import os

# outpurt directory
os.makedirs("output", exist_ok=True)

output_path = "output/lane_pothole_output.mp4"

fps_input = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps_input,
                      (frame_width, frame_height))


prev_time = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_resized = cv2.resize(frame, (640, 640))

    results_seg = model_lane.predict(source=frame_resized, imgsz=640, conf=0.3, device=device, verbose=False)

    if len(results_seg) > 0 and results_seg[0].masks is not None:
        masks = results_seg[0].masks.data.cpu().numpy()

        # Resize masks to original frame size
        h, w, _ = frame.shape
        lane_masks = [cv2.resize(mask, (w, h)) for mask in masks]

        overlay = frame.copy()

        for mask in lane_masks:
            mask_bin = (mask > 0.5).astype(np.uint8)

            # Green color for all lanes
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[:, :, 1] = mask_bin * 255

            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

        frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    results_det = model_detect.predict(source=frame_resized, imgsz=640, conf=0.25, device=device, verbose=False)[0]

    for box in results_det.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = results_det.names[cls_id]

        color = (0, 0, 255) if "pothole" in label.lower() or "speed" in label.lower() else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out.write(frame)

    cv2.imshow("YOLOv8 Lanes + Pothole Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
