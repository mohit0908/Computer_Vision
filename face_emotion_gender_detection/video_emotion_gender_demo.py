from statistics import mode
import keras
import imutils
import cv2
from keras.models import load_model
import numpy as np
from imutils.video import VideoStream
import time
import os
from keras.utils.generic_utils import CustomObjectScope
from preprocessor import preprocess_input

# Support functions

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif dataset_name == 'KDEF':
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    else:
        raise Exception('Invalid dataset name')


def detect_faces(detection_model, gray_image_array, conf):
    frame = gray_image_array
    # Grab frame dimention and convert to blob
    (h,w) =  frame.shape[:2]
    # Preprocess input image: mean subtraction, normalization
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))
    # Set read image as input to model
    detection_model.setInput(blob)

    # Run forward pass on model. Receive output of shape (1,1,no_of_predictions, 7)
    predictions = detection_model.forward()
    coord_list = []
    for i in range(0, predictions.shape[2]):
        confidence = predictions[0,0,i,2]
        if confidence > conf:
            # Find box coordinates rescaled to original image
            box_coord = predictions[0,0,i,3:7] * np.array([w,h,w,h])
            conf_text = '{:.2f}'.format(confidence)
            # Find output coordinates
            xmin, ymin, xmax, ymax = box_coord.astype('int')
            coord_list.append([xmin, ymin, (xmax-xmin), (ymax-ymin)])
    print('Coordinate list:', coord_list)
    return coord_list


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def load_detection_model(prototxt, weights):
    detection_model = cv2.dnn.readNetFromCaffe(prototxt, weights)
    return detection_model


# parameters for loading data and images
prototxt = 'trained_models/deploy.prototxt.txt'
weights = 'trained_models/res10_300x300_ssd_iter_140000.caffemodel'
emotion_model_path = 'trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = 'trained_models/genderMobilenet_finetuned_alphahalf6sept.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)
confidence = 0.6

# loading models
face_detection = load_detection_model(prototxt, weights)
emotion_classifier = load_model(emotion_model_path, compile=False)

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    gender_classifier = load_model(gender_model_path, compile=False)


# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
emotion_window = []

filepath = 'Video'
input_filename = 'friends.mp4'
output_path = os.path.join(filepath, input_filename.split('.')[0]+'_output.avi')

counter = 0
frame_process_counter = 0
# starting video streaming
cv2.namedWindow('window_frame')
input_video = os.path.join(filepath, input_filename)
video_capture = cv2.VideoCapture(input_video)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path,fourcc, 15.0, (640,480))
time.sleep(1.0)



while (video_capture.isOpened()):
    ret, bgr_image = video_capture.read()
    counter += 1
    # Edit this counter value to process every nth frame
    if counter % 3 == 0:
        frame_process_counter += 1
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, bgr_image,confidence)
        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
            rgb_face = rgb_image[y1:y2, x1:x2]

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                rgb_face = cv2.resize(rgb_face, (gender_target_size))
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue
            gray_face = preprocess_input(gray_face, False)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            rgb_face = np.expand_dims(rgb_face, 0)
            rgb_face = preprocess_input(rgb_face, False)
            gender_prediction = gender_classifier.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]
            gender_window.append(gender_text)

            if len(gender_window) > frame_window:
                emotion_window.pop(0)
                gender_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
                gender_mode = mode(gender_window)
            except:
                continue

            if gender_text == gender_labels[0]:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, gender_mode,
                      color, 0, -20, 1, 1)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        bgr_image = cv2.resize(bgr_image, (640, 480))
        out.write(bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Total frames:', counter)
        print('Frames processed:', frame_process_counter)
        print('Frame processing ratio:{:.2f} %'.format((np.float(frame_process_counter)/counter)*100))
        break
video_capture.release()
out.release()
cv2.destroyAllWindows()