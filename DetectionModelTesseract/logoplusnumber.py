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
# Tesseract Path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Download YOLO Brand Model
model_path = hf_hub_download(
    repo_id="haydarkadioglu/brand-eye",
    filename="brandeye.pt"
)

model = YOLO(model_path)

# Use ONE camera stream (change if needed)
IP_CAM_URL = "http://192.168.29.68:4747/video"

cap = cv2.VideoCapture(IP_CAM_URL)

if not cap.isOpened():
    print("Failed to open IP camera.")
    exit()

time.sleep(2)

last_plate = ""
detected_brand = None
detected_plate = None


# ------------------- NUMBER PLATE FUNCTION -------------------

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
    print("Insert detection Successfully")

def detect_plate_color(plate_img):
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)

    # Define HSV ranges
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
        box = cv2.boxPoints(rect)          # float32 from OpenCV
        box = box.astype(np.intp)          # was np.int0, now explicit int type

        w = rect[1][0]
        h = rect[1][1]

        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / min(w, h)

        # Looser aspect ratio; adjust if needed
        if 1.0 < aspect_ratio < 10.0:
            # Order points: tl, tr, br, bl
            src_pts = box.astype("float32")   # keep float32 for transform
            src_pts = sorted(src_pts, key=lambda x: x[0])
            left = src_pts[:2]
            right = src_pts[2:]
            left = sorted(left, key=lambda x: x[1])
            right = sorted(right, key=lambda x: x[1])
            tl, bl = left
            tr, br = right

            width = int(max(w, h))
            height = int(min(w, h))

            # Skip too small regions
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
        print("Detected Plate Color:", plate_color)
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)

        # Try Otsu threshold first (often better for plates)
        _, thresh = cv2.threshold(
            plate_gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Tesseract config: single text line, uppercase letters and digits
        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        raw_text = pytesseract.image_to_string(thresh, config=config)
        print("RAW OCR:", repr(raw_text))

        text = "".join([c for c in raw_text if c.isalnum()]).upper()

        cv2.imshow("Plate ROI", plate_img)
        cv2.imshow("Plate Thresh", thresh)

        if text:
            return text,plate_color

    return None,None


# ------------------- MAIN LOOP -------------------

print("Waiting for both Logo and Number Plate detection...\n")

indian_states = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "DL": "Delhi",
    "JK": "Jammu and Kashmir",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "PY": "Puducherry"
}

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (600, 400))

  #firstly we are detecting the logo of the car
    if detected_brand is None:
        results = model(frame, conf=0.3)
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            detected_brand = model.names[int(boxes[0].cls[0])]
            print("Detected Brand:", detected_brand)

    # detecting the number plate of the car
    if detected_plate is None:
        plate_text,plate_color = detect_plate_and_info(frame)

        if plate_text:
            detected_plate = plate_text
            detected_color=plate_color
            state_code=plate_text[0:2]
            if state_code is not None and state_code in indian_states.keys() and len(detected_plate)==10:
                detected_state_code=indian_states[state_code]
            else:
                break
            # print("Detected Plate:", detected_plate)
            # print("Plate Color:",detected_color)
            # print("State:",detected_state_code)
    if detected_brand is not None and detected_plate is not None and detected_color is not None and len(detected_plate)==10:
        # print("\n BOTH DETECTED SUCCESSFULLY")

        # print("Brand:", detected_brand)
        # print("Plate:", detected_plate)
        # print("Color:",detected_color)
        # print("State:",detected_state_code)
        vehicleType="Not detected"
        if detected_color=="White":
            vehicleType="Private"
        elif detected_color=="Yellow":
            vehicleType="Taxi"
        elif detected_color=="Green":
            vehicleType="EV"
        else:
            vehicleType="Distinguish"

        insert_into_db(detected_plate,detected_color,vehicleType,detected_brand)
        break

    #cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
