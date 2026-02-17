import cv2
import numpy as np
import imutils
import pytesseract
import time


# ------------------- CONFIG -------------------

# Path to Tesseract executable (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Print version to confirm Tesseract is found
try:
    print("Tesseract version:", pytesseract.get_tesseract_version())
except Exception as e:
    print("ERROR: Tesseract not found or path wrong:", e)


# Your IP camera / phone camera URL
IP_CAM_URL = "http://192.168.29.68:4747/video"


# ------------------- OPEN CAMERA -------------------

cap = cv2.VideoCapture(IP_CAM_URL)

print("Camera opened:", cap.isOpened())
if not cap.isOpened():
    print("Failed to open IP camera stream. Check URL / network / app settings.")
    exit()

# Optional: small delay to let stream stabilize
time.sleep(2)

last_plate = ""


# ------------------- PLATE DETECTION FUNCTION -------------------

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
            return text

    return None


# ------------------- MAIN LOOP -------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("IP Camera frame read failed!")
        break

    # Resize for speed / consistency
    frame = cv2.resize(frame, (900, 600))

    plate_text = detect_plate_and_info(frame)

    if plate_text and plate_text != last_plate and len(plate_text) >= 6:
        last_plate = plate_text
        if plate_text is not None:
            print("Detected Plate:", plate_text)
            break

    cv2.imshow("IP Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
