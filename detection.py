import cv2
import imutils

#reading the image from the source
image=cv2.imread('car.jpg')

#if the image size is smaller and system is not able to recognize the number plate
#then it will be very difficult to detect the number plate of car.
#image=imutils.resize(img, width=600)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),0)

edges=cv2.Canny(blur,50,150)

cnts=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
cnts=sorted(cnts,key=cv2.contourArea, reverse=True)[:10]

plate=None

for c in cnts:
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.018*peri, True)

    if len(approx)==4: #rectangle shape
        plate=approx
        break

if plate is not None:
    cv2.drawContours(image, [plate],-1,(0,255,0),3)
    
        

#displaying the image using imshow command

cv2.imshow("Number Plate Detection",image)
cv2.waitKey(0)
cv2.destroyAllWindows()