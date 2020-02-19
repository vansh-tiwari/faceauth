from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2 as cv
import os

saveFile = "vansh"
outputPath = "datasetImages/"+saveFile
cascadeFile = 'haarcascade_frontalface_default.xml'

detector = cv.CascadeClassifier(cascadeFile)

print("==> Working on camera...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

while True:
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(
		cv.cvtColor(frame, cv.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

	for (x, y, w, h) in rects:
		cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# p = os.path.sep.join([outputPath, "{}.png".format(str(total).zfill(5))])
		# cv.imwrite(p, orig)
		# total += 1

	cv.imshow("Frame", frame)
	key = cv.waitKey(1) & 0xFF
	saveName = saveFile+"_"+str(total)
	if key == ord("k"):
		p = os.path.sep.join([outputPath, "{}.jpg".format(saveName.zfill(5))])
		cv.imwrite(p, orig)
		total += 1

	if key == ord("q"):
		break

print("==> {} face images saved".format(total))
cv.destroyAllWindows()
vs.stop()