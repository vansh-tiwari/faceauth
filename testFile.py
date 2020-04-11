import os
import cv2 as cv
import numpy as np
import recognition as rc

img_test = cv.imread('testImages/i.jpg')
facesDetected, gray_image = rc.Detect(img_test)
print("==> Detected Faces: ", facesDetected)

# for (x,y,w,h) in facesDetected:
#     cv.rectangle(img_test,(x,y),(x+w+5,y+h+5),(120,125,255),thickness=2)

# resized = cv.resize(img_test,(1000,700))
# cv.imshow("Detected Face", resized)
# cv.waitKey(0)
# cv.destroyAllWindows

faces, faceID = rc.labelsTrainData('trainImages')
faceRecognizer = rc.trainClassifier(faces, faceID)
# faceRecognizer.save('trainingData.yml')

nameDict = {1: 'Vansh', 2: 'Felicity', 3: 'Arrow', 4:'Alex'}

for face in facesDetected:
    (x,y,w,h) = face
    roiGray = gray_image[y:y+h, x:x+h]
    label, confidence = faceRecognizer.predict(roiGray)
    # if confidence > 35:
    print("Confidence: {}, Label: {}".format(confidence,label))
    rc.drawRect(img_test, face)
    predictedName = nameDict[label]
    rc.keepText(img_test, predictedName, x, y)
    # else:print("Need more accurate dataset")

resized = cv.resize(img_test,(1000,700))
cv.imshow("Detected Face", resized)
cv.waitKey(0)
cv.destroyAllWindows