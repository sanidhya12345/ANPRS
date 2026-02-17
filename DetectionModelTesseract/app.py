import cv2
import numpy as np
import imutils
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Laptop webcam
cap = cv2.VideoCapture(0)

STATE_CODES = {
    "UP": "Uttar Pradesh",
    "DL": "Delhi",
    "MH": "Maharashtra",
    "HR": "Haryana",
    "RJ": "Rajasthan",
    "PB": "Punjab",
    "UK": "Uttarakhand",
    "MP": "Madhya Pradesh",
    "BR": "Bihar",
    "GJ": "Gujarat",
    "WB": "West Bengal",
    "TN": "Tamil Nadu",
    "KA": "Karnataka",
    "AP": "Andhra Pradesh",
    "TS": "Telangana",
    "KL": "Kerala"
}

last_plate = ""


def detect_plate_and_info(frame):
    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)

    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]

    plate_img = None

    for c in cnts:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        w = rect[1][0]
        h = rect[1][1]

        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / min(w, h)

        if 2 < aspect_ratio < 6:
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

            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst_pts)
            warped = cv2.warpPerspective(orig, M, (width, height))

            plate_img = warped
            break

    if plate_img is not None:
        # Plate Color Detection
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        h, s, v = avg_hsv

        plate_color = "Unknown"
        if v > 180 and s < 50:
            plate_color = "White"
        elif h < 40 and s > 80:
            plate_color = "Yellow"
        elif 35 < h < 85 and s > 80:
            plate_color = "Green"
        else:
            plate_color = "Other"

        # OCR
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)

        thresh = cv2.adaptiveThreshold(
            plate_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        text = pytesseract.image_to_string(thresh, config="--psm 7")
        text = "".join([c for c in text if c.isalnum()]).upper()

        state_code = text[:2] if len(text) >= 2 else ""
        state_name = STATE_CODES.get(state_code, "Unknown State")

        return text, plate_color, state_name

    return None, None, None


while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working!")
        break

    frame = cv2.resize(frame, (900, 600))

    plate_text, plate_color, state_name = detect_plate_and_info(frame)

    if plate_text and plate_text != last_plate and len(plate_text) >= 4:
        last_plate = plate_text
        print("\n==============================")
        print("Detected Plate:", plate_text)
        print("Plate Color:", plate_color)
        print("State:", state_name)
        print("==============================")

    cv2.imshow("Laptop Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.5)  # avoid printing too fast

cap.release()
cv2.destroyAllWindows()
