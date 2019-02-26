import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.video import VideoStream
import time


def find_faces(prototxt, weights, conf):

	# Read Caffe model
	model = cv2.dnn.readNetFromCaffe(prototxt, weights)
	# Start video stream and warmup webcam
	videostrean = VideoStream(src = 0).start()
	time.sleep(1.0)

	while True:
		frame = videostrean.read()
		frame = imutils.resize(frame, width = 800)
		# Grab frame dimention and convert to blob
		(h,w) =  frame.shape[:2]
		# Preprocess input image: mean subtraction, normalization

		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
		
		# Set read image as input to model
		model.setInput(blob)

		# Run forward pass on model. Receive output of shape (1,1,no_of_predictions, 7)
		predictions = model.forward()

		for i in range(0, predictions.shape[2]):
			confidence = predictions[0,0,i,2]
			if confidence > conf:
				# Find box coordinates rescaled to original image
				box_coord = predictions[0,0,i,3:7] * np.array([w,h,w,h])
				conf_text = '{:.2f}'.format(confidence)
				# Find output coordinates
				xmin, ymin, xmax, ymax = box_coord.astype('int')
				print('confidence:', confidence*100)

				# Draw bounding boxes on input image
				cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
					(255,0,2), 2)
				# Write confidence value on top of bounding box
				cv2.putText(frame, conf_text, (xmin, ymin),
					cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)

	# Display image
		cv2.imshow('Frame',frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
	cv2.destroyAllWindows()
	videostrean.stop()


if __name__ == '__main__':

	# Parse command line arguments

	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--prototxt', required = True,
		help = 'path to caffe model architecture file')
	parser.add_argument('-w', '--weights', required = True,
		help = 'path to model weights file to be loaded in model architecture')
	parser.add_argument('-c', '--confidence', type = float, 
		help = 'probability threshold for detection', default = 0.3)

	args = vars(parser.parse_args())

	# Call object prediction method
	find_faces(args['prototxt'], args['weights'],args['confidence'])


# Usage: python detect_faces_video.py --prototxt deploy.prototxt.txt --weights res10_300x300_ssd_iter_140000.caffemodel --confidence 0.6
