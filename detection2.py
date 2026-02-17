import cv2
import pytesseract
import numpy as np
import imutils
import re

# If on Windows, set path to tesseract.exe, e.g.:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INDIAN_PLATE_REGEX = r"^[A-Z]{2}[ -]?[0-9]{2}[ -]?[A-Z]{1,2}[ -]?[0-9]{4}$"

def clean_indian_plate(text):
    text = "".join(ch for ch in text.upper() if ch.isalnum())
    if re.match(INDIAN_PLATE_REGEX, text):
        return text
    return text

def detect_plate_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    plate_contour = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is None:
        return None, None

    x, y, w, h = cv2.boundingRect(plate_contour)
    plate_img = image[y:y + h, x:x + w]

    return plate_img, (x, y, w, h)

def recognize_indian_plate_cpu_tesseract(image_path, show_steps=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image, check the path")

    img = imutils.resize(img, width=800)

    plate_img, bbox = detect_plate_region(img)
    if plate_img is None:
        print("No plate region detected.")
        return None

    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
    _, plate_thresh = cv2.threshold(
        plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if show_steps:
        cv2.imshow("Plate", plate_img)
        cv2.imshow("Plate Thresh", plate_thresh)
        cv2.waitKey(0)

    # Tesseract config: treat image as single text line, whitelist characters
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(plate_thresh, config=config)
    cleaned = clean_indian_plate(text)

    if bbox is not None:
        (x, y, w, h) = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img, cleaned, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    if show_steps:
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cleaned

if __name__ == "__main__":
    image_path = "car.jpg"
    plate = recognize_indian_plate_cpu_tesseract(image_path, show_steps=True)
    print("Detected Indian plate:", plate)
