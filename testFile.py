import os
import cv2 as cv
import numpy as np
import recognition as rc

img_test = cv.imread('testImages/test1.png')
faces, gray_image = rc.Detect(img_test)
print("==> Detected Faces: ", faces)

for (x,y,w,h) in faces:
    cv.rectangle(img_test,(x,y),(x+w+5,y+h+5),(120,125,255),thickness=2)

resized = cv.resize(img_test,(1000,700))
cv.imshow("Detected Face", resized)
cv.waitKey(0)
cv.destroyAllWindows