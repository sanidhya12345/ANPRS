import cv2
import imutils
import pytesseract

# For this your system must have Tesseract-OCR->tesseract.exe file in your system.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread("car.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, 50, 150)

cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

plate = None
plate_roi = None

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)

    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        if 2 < aspect_ratio < 6:
            plate = approx
            plate_roi = gray[y:y+h, x:x+w]
            break

if plate_roi is not None:
    # preprocessing for OCR
    plate_roi = cv2.bilateralFilter(plate_roi, 11, 17, 17)
    _, thresh = cv2.threshold(plate_roi, 150, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(thresh, config="--psm 8")

    print("Number Plate Text:", text.strip())

    cv2.imshow("Extracted Plate", thresh)
    cv2.waitKey(0)

else:
    print("Number plate not detected!")
