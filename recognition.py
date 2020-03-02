import face_recognition as fr
import cv2 as cv
import os
import numpy as np

def Detect(test_img):
    gray_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

    return faces, gray_img

def labelsTrainData(dir):
    faces = []
    faceID = []
    for path, subdir, files in os.walk(dir):
        for file in files:
            if file.startswith("."):
                print("SystemFile")
                continue
            id = os.path.basename(path)
            imgPath = os.path.join(path, file)
            print("imgPath -> ", imgPath)
            print("id -> ",id)
            testImg = cv.imread(imgPath)
            if testImg is None:
                print("Error Loading Image")
                continue
            rectFaces, grayImg = Detect(testImg)
            if len(rectFaces)!=1:
                continue #Face should be one

            (x,y,w,h) = rectFaces[0]
            
            #RegionOfInterest
            roiGray = grayImg[y:y+w,x:x+h]
            faces.append(roiGray)
            faceID.append(int(id))
    return faces, faceID

def trainClassifier(faces,faceID):
    faceRecognizer = cv.face.LBPHFaceRecognizer_create()
    faceRecognizer.train(faces, np.array(faceID))
    return faceRecognizer

def drawRect(testImg, face):
    (x,y,w,h) = face
    cv.rectangle(testImg,(x,y),(x+w+5,y+h+5),(120,125,255),thickness=2)

def keepText(testImg, text, x, y):
    cv.putText(testImg, text, (x,y), cv.FONT_HERSHEY_DUPLEX, 5, (255,0,0), 6)