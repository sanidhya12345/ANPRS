import cv2
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Download model
model_path = hf_hub_download(
    repo_id="haydarkadioglu/brand-eye",
    filename="brandeye.pt"
)

# Load model
model = YOLO(model_path)
def detect_brands(image_path, conf_threshold=0.25):
    """
    Detect brands in a single image
    
    Args:
        image_path (str): Path to the image file
        conf_threshold (float): Confidence threshold (0.0-1.0)
    
    Returns:
        results: Detection results with bounding boxes and labels
    """
    results = model(image_path, conf=conf_threshold)
    
    # Display results
    results[0].show()
    
    # Get detection details
    boxes = results[0].boxes
    if boxes is not None:
        print(f"Found {len(boxes)} brand detections:")
        for box in boxes:
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = model.names[cls]
            print(f"  - {class_name}: {conf:.3f} confidence")
    
    return results

# Example usage
results = detect_brands("D:\ANPRS\car_plate_images\car5.jpg")
