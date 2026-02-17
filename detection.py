import cv2
import numpy as np
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread("car.jpg")
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, 50, 200)

cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]

plate_img = None

for c in cnts:
    rect = cv2.minAreaRect(c)   # rotated rectangle
    box = cv2.boxPoints(rect)
    box = box.astype(int)



    w = rect[1][0]
    h = rect[1][1]

    if w == 0 or h == 0:
        continue

    aspect_ratio = max(w, h) / min(w, h)

    # plate ratio generally 2 to 6
    if 2 < aspect_ratio < 6:
        # perspective transform
        src_pts = box.astype("float32")

        # reorder points
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
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(plate_gray, 255,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2)


    text = pytesseract.image_to_string(thresh, config="--psm 7")
    print("Detected Plate Text:", text.strip())

    cv2.imshow("Extracted Plate", plate_img)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)
else:
    print("Number plate not detected!")
