import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Download model
model_path = hf_hub_download(
    repo_id="haydarkadioglu/brand-eye",
    filename="brandeye.pt"
)

# Load model
model = YOLO(model_path)


def real_time_detection():

    IP_CAM_URL = "http://192.168.29.68:4747/video"
    cap = cv2.VideoCapture(IP_CAM_URL)

    if not cap.isOpened():
        print("Error: Cannot open IP camera")
        return

    print("Waiting for detection...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Run detection
        results = model(frame, conf=0.3)
        boxes = results[0].boxes

        # If something detected → stop everything
        if boxes is not None and len(boxes) > 0:
            class_name = model.names[int(boxes[0].cls[0])]
            print("Detected Brand:", class_name)
            break   # ✅ This breaks the while loop

    cap.release()
    cv2.destroyAllWindows()
    print("Detection complete. Program stopped.")


real_time_detection()