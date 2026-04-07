import cv2
import numpy as np
import imutils
import pytesseract
import time
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from pymongo import MongoClient
from datetime import datetime

# ------------------- CONFIG -------------------

MONGO_URI="mongodb+srv://varshneysanidhya_db_user:IdlnLOhPaoW7XC6r@cluster0.1z1jtnl.mongodb.net/?appName=Cluster0"

client=MongoClient(MONGO_URI)
db=client['vehicledb']
collection=db['detectiondata']

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

model_path = hf_hub_download(
    repo_id="haydarkadioglu/brand-eye",
    filename="brandeye.pt"
)

model = YOLO(model_path)

IP_CAM_URL = "http://192.168.29.68:4747/video"
cap = cv2.VideoCapture(IP_CAM_URL)

if not cap.isOpened():
    print("Failed to open IP camera.")
    exit()

time.sleep(2)

# ------------------- VARIABLES -------------------

detected_brand = None
detected_plate = None
detected_color = None

last_insert_time = 0
cooldown = 5  # seconds

# ------------------- DB INSERT -------------------

def insert_into_db(plate_number, color, vehicle_type, brand):

    vehicle_data = {
        "number": plate_number,
        "color": color,
        "vehicleType": vehicle_type,
        "brandName": brand,
        "location": "Gate-1",
        "timestamp": datetime.now()
    }

    collection.insert_one(vehicle_data)
    print("✅ Inserted:", plate_number)

# ------------------- COLOR DETECTION -------------------

def detect_plate_color(plate_img):
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "White":  [(0, 0, 180), (180, 40, 255)],
        "Yellow": [(15, 80, 80), (40, 255, 255)],
        "Green":  [(35, 50, 50), (85, 255, 255)],
        "Black":  [(0, 0, 0), (180, 255, 50)],
        "Red":    [(0, 70, 50), (10, 255, 255)]
    }

    max_pixels = 0
    detected_color = "Unknown"

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        pixels = cv2.countNonZero(mask)

        if pixels > max_pixels:
            max_pixels = pixels
            detected_color = color

    return detected_color

# ------------------- PLATE DETECTION -------------------

def detect_plate_and_info(frame):
    orig = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)

    cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    plate_img = None

    for c in cnts:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = box.astype(np.intp)

        w = rect[1][0]
        h = rect[1][1]

        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / min(w, h)

        if 1.0 < aspect_ratio < 10.0:
            src_pts = box.astype("float32")
            src_pts = sorted(src_pts, key=lambda x: x[0])

            left = src_pts[:2]
            right = src_pts[2:]

            left = sorted(left, key=lambda x: x[1])
            right = sorted(right, key=lambda x: x[1])

            tl, bl = left
            tr, br = right

            width = int(max(w, h))
            height = int(min(w, h))

            if width < 80 or height < 20:
                continue

            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(
                np.array([tl, tr, br, bl], dtype="float32"),
                dst_pts
            )

            warped = cv2.warpPerspective(orig, M, (width, height))
            plate_img = warped
            break

    if plate_img is not None:
        plate_color = detect_plate_color(plate_img)

        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)

        _, thresh = cv2.threshold(
            plate_gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        raw_text = pytesseract.image_to_string(thresh, config=config)

        text = "".join([c for c in raw_text if c.isalnum()]).upper()

        cv2.imshow("Plate ROI", plate_img)
        cv2.imshow("Plate Thresh", thresh)

        if text:
            return text, plate_color

    return None, None

# ------------------- STATES -------------------

indian_states = {
    "UP": "Uttar Pradesh",
    "DL": "Delhi",
    "MH": "Maharashtra",
    "HR": "Haryana"
}

# ------------------- MAIN LOOP -------------------

print("🚀 Detection Started...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (600, 400))

    # 🔥 Brand detection
    if detected_brand is None:
        results = model(frame, conf=0.3)
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            detected_brand = model.names[int(boxes[0].cls[0])]
            print("Brand:", detected_brand)

    # 🔥 Plate detection
    if detected_plate is None:
        plate_text, plate_color = detect_plate_and_info(frame)

        if plate_text:
            if len(plate_text) == 10 and plate_text[:2] in indian_states:
                detected_plate = plate_text
                detected_color = plate_color
                print("Plate:", detected_plate, "| Color:", detected_color)
            else:
                detected_plate = None
                continue

    # 🔥 Final condition + cooldown
    current_time = time.time()

    if (detected_brand and detected_plate and detected_color):

        if current_time - last_insert_time > cooldown:

            vehicleType = "Not detected"
            if detected_color == "White":
                vehicleType = "Private"
            elif detected_color == "Yellow":
                vehicleType = "Taxi"
            elif detected_color == "Green":
                vehicleType = "EV"
            else:
                vehicleType = "Other"

            insert_into_db(
                detected_plate,
                detected_color,
                vehicleType,
                detected_brand
            )

            last_insert_time = current_time

        # 🔁 RESET for next detection
        detected_brand = None
        detected_plate = None
        detected_color = None

    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()