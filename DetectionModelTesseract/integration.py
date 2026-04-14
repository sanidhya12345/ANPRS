from flask import Flask, render_template, request, jsonify, Response
from pymongo import MongoClient
from datetime import datetime
from collections import Counter
import threading
import cv2
import numpy as np
import imutils
import pytesseract
import time
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

app = Flask(__name__)

# ------------------- CONFIG & SETUP -------------------

# MongoDB Setup
MONGO_URI = "mongodb+srv://varshneysanidhya_db_user:IdlnLOhPaoW7XC6r@cluster0.1z1jtnl.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client['vehicledb']
collection = db['detectiondata']
blacklist_collection = db['blacklist']
alerts_collection = db['alerts']



# Tesseract Setup (Update path if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# YOLO Setup
model_path = hf_hub_download(
    repo_id="haydarkadioglu/brand-eye",
    filename="brandeye.pt"
)
model = YOLO(model_path)

# ------------------- GLOBAL VARIABLES -------------------
latest_frame = None
detected_brand = None
detected_plate = None
detected_color = None

# New Cooldown Logic
vehicle_last_seen = {}
VEHICLE_COOLDOWN = 120  # 120 seconds = 2 minutes

indian_states = {
    "UP": "Uttar Pradesh",
    "DL": "Delhi",
    "MH": "Maharashtra",
    "HR": "Haryana"
}

# ------------------- CAMERA THREADING (LAG FIX) -------------------
class CameraReader:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2) 
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                time.sleep(0.01)
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True

IP_CAM_URL = "http://10.211.10.208:4747/video"
#IP_CAM_URL = "http://10.211.10.126:5000/video_feed"
print("Connecting to IP Camera...")
cam_reader = CameraReader(IP_CAM_URL).start()
time.sleep(2) # Warm up time

# ------------------- HELPER FUNCTIONS -------------------

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
        w, h = rect[1][0], rect[1][1]

        if w == 0 or h == 0: continue
        aspect_ratio = max(w, h) / min(w, h)

        if 1.0 < aspect_ratio < 10.0:
            src_pts = sorted(box.astype("float32"), key=lambda x: x[0])
            left = sorted(src_pts[:2], key=lambda x: x[1])
            right = sorted(src_pts[2:], key=lambda x: x[1])
            tl, bl = left
            tr, br = right
            width, height = int(max(w, h)), int(min(w, h))

            if width < 80 or height < 20: continue

            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst_pts)
            plate_img = cv2.warpPerspective(orig, M, (width, height))
            break

    if plate_img is not None:
        plate_color = detect_plate_color(plate_img)
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
        _, thresh = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        raw_text = pytesseract.image_to_string(thresh, config=config)
        text = "".join([c for c in raw_text if c.isalnum()]).upper()

        if text:
            return text, plate_color

    return None, None

# ------------------- BACKGROUND DETECTION TASK -------------------
def detection_task():
    global latest_frame, detected_brand, detected_plate, detected_color, vehicle_last_seen

    print("Background Detection Started...\n")
    frame_count = 0 

    while True:
        ret, frame = cam_reader.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (600, 400))
        latest_frame = frame.copy() # Save for Flask Video Stream
        frame_count += 1
        
        # Process every 3rd frame to save CPU
        if frame_count % 3 != 0:
            continue

        # Brand detection
        if detected_brand is None:
            results = model(frame, conf=0.3, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                detected_brand = model.names[int(boxes[0].cls[0])]

        # Plate detection
        if detected_plate is None:
            plate_text, plate_color = detect_plate_and_info(frame)
            if plate_text and len(plate_text) == 10 and plate_text[:2] in indian_states:
                detected_plate = plate_text
                detected_color = plate_color

        # Final condition + 2-Minute Cooldown Logic
        current_time = time.time()
        if detected_brand and detected_plate and detected_color:
            
            last_seen = vehicle_last_seen.get(detected_plate, 0)
            
            if current_time - last_seen > VEHICLE_COOLDOWN:
                vehicleType = "Not detected"
                if detected_color == "White": vehicleType = "Private"
                elif detected_color == "Yellow": vehicleType = "Taxi"
                elif detected_color == "Green": vehicleType = "EV"
                else: vehicleType = "Other"

                insert_into_db(detected_plate, detected_color, vehicleType, detected_brand)
                vehicle_last_seen[detected_plate] = current_time # Update cooldown
            else:
                pass # Skipping duplicate

            #  RESET for next detection
            detected_brand = None
            detected_plate = None
            detected_color = None

# ------------------- FLASK ROUTES -------------------
def gen_frames():
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode('.jpg', latest_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05) 

@app.route('/', methods=['GET'])
def dashboard():
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    vehicle_type = request.args.get('vehicleType')
    location = request.args.get('location')

    query = {}
    if start_date and end_date:
        query["timestamp"] = {"$gte": datetime.fromisoformat(start_date), "$lte": datetime.fromisoformat(end_date)}
    if vehicle_type:
        query["vehicleType"] = vehicle_type
    if location:
        query["location"] = location

    if collection.count_documents({}) == 0:
        alerts_collection.delete_many({})

    data = list(collection.find(query))
    total = len(data)
    
    hours = [d.get("timestamp").hour for d in data if d.get("timestamp")]
    hourly = Counter(hours)
    alerts = list(alerts_collection.find().sort("timestamp", -1).limit(5))
    
    # Fetch recent 20 for the table
    recent_detections = list(collection.find(query).sort("timestamp", -1).limit(20))

    return render_template(
        "dashboard.html",
        total=total,
        hourly=hourly,
        alerts=alerts,
        recent_detections=recent_detections
    )

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime-data')
def realtime_data():
    data = list(collection.find())
    total = len(data)

    if total == 0: alerts_collection.delete_many({})

    hours = [d.get("timestamp").hour for d in data if d.get("timestamp")]
    hourly = dict(Counter(hours))

    latest_vehicle = collection.find_one(sort=[("_id", -1)])
    alert = None

    # Alert Checking Logic
    if latest_vehicle:
        latest_plate = latest_vehicle.get("number")
        black = blacklist_collection.find_one({"number": latest_plate})

        if black:
            last_alert = alerts_collection.find_one(sort=[("timestamp", -1)])
            if not last_alert or last_alert.get("number") != latest_plate:
                alert_reason = black.get("reason", "Blacklisted")
                alerts_collection.insert_one({
                    "number": latest_plate, "reason": alert_reason, "timestamp": datetime.now()
                })
                alert = {"number": latest_plate, "reason": alert_reason}

    alerts_list = list(alerts_collection.find().sort("timestamp", -1).limit(5))
    for a in alerts_list:
        a['_id'] = str(a['_id'])
        a['timestamp'] = a['timestamp'].isoformat() if 'timestamp' in a else ""
        
    recent_list = list(collection.find().sort("timestamp", -1).limit(20))
    for r in recent_list:
        r['_id'] = str(r['_id'])
        r['timestamp'] = r['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if 'timestamp' in r else ""

    return jsonify({
        "total": total,
        "hourly": hourly,
        "alerts_list": alerts_list,
        "alert": alert,
        "recent_detections": recent_list
    })

# ------------------- RUN APP -------------------
if __name__ == "__main__":
    t = threading.Thread(target=detection_task, daemon=True)
    t.start()
    app.run(debug=True, use_reloader=False)