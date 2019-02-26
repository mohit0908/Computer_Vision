import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt


def find_faces(prototxt, weights, image, conf):

	# Read Caffe model
	model = cv2.dnn.readNetFromCaffe(prototxt, weights)
	# Read input image
	input_img = plt.imread(image)
	# input_img = np.expand_dims(input_img, axis = 0)
	# print(input_img.shape)


	(h,w) =  input_img.shape[:2]
	# Preprocess input image: mean subtraction, normalization

	blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
	# input_img = input_img.reshape(1,3,300,300)
	
	# Set read image as input to model

	model.setInput(blob)

	# Run forward pass on model. Receive output of shape (1,1,no_of_predictions, 7)
	predictions = model.forward()
	# print('predictions:', predictions.shape)

	for i in range(0, predictions.shape[2]):
		confidence = predictions[0,0,i,2]
		if confidence > conf:
			print('Optimum confidence found')
			# Find box coordinates rescaled to original image
			box_coord = predictions[0,0,i,3:7] * np.array([w,h,w,h])
			conf_text = '{:.2f}'.format(confidence)
			# print(conf_text)
			# Find output coordinates
			xmin, ymin, xmax, ymax = box_coord.astype('int')
			print('confidence:', confidence*100)

			# Draw bounding boxes on input image
			cv2.rectangle(input_img, (xmin, ymin), (xmax, ymax),
				(255,0,2), 2)
			# Write confidence value on top of bounding box
			cv2.putText(input_img, conf_text, (xmin, ymin),
				cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)

	# Display image
	plt.imshow(input_img)
	plt.show()



if __name__ == '__main__':

	# Parse command line arguments

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--image', required = True,
		help = 'path to inference image')
	parser.add_argument('-p', '--prototxt', required = True,
		help = 'path to caffe model architecture file')
	parser.add_argument('-w', '--weights', required = True,
		help = 'path to model weights file to be loaded in model architecture')
	parser.add_argument('-c', '--confidence', type = float, 
		help = 'probability threshold for detection', default = 0.3)

	args = vars(parser.parse_args())

	# Call object prediction method
	find_faces(args['prototxt'], args['weights'], args['image'], args['confidence'])


# Usage: python detect_faces.py --image images/people.jpg --prototxt deploy.prototxt.txt --weights res10_300x300_ssd_iter_140000.caffemodel --confidence 0.6