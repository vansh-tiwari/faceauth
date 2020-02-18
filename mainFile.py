import face_recognition as fr
import cv2 as cv
import os
import numpy as np

def Detect(test_img):
    gray_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

    return faces, gray_img