from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2 as cv
import os

outputPath = "datasetImages/vansh"
cascadeFile = 'haarcascade_frontalface_default.xml'

detector = cv.CascadeClassifier(cascadeFile)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0
saveFile = "vansh"

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, clone it, (just
	# in case we want to write it to disk), and then resize the frame
	# so we can apply face detection faster
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(
		cv.cvtColor(frame, cv.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

	# loop over the face detections and draw them on the frame
	for (x, y, w, h) in rects:
		cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# p = os.path.sep.join([outputPath, "{}.png".format(str(total).zfill(5))])
		# cv.imwrite(p, orig)
		# total += 1

	# show the output frame
	cv.imshow("Frame", frame)
	key = cv.waitKey(1) & 0xFF
	
	# if the `k` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == ord("k"):
		p = os.path.sep.join([outputPath, "{}.png".format(str(total).zfill(5))])
		cv.imwrite(p, orig)
		total += 1

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv.destroyAllWindows()
vs.stop()