import cv2

from cnn.preprocessor import Preprocessor
from utils import image_utils
from utils.image_utils import crop, draw_rectangle, detect_face, show_cv2_img
import numpy as np


def camera_smile_detection(model):
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cor, found = detect_face(frame)

        if found:
            face = image_utils.crop(frame, cor)
            face = cv2.resize(face, (64, 64))
            input_data = np.expand_dims(face, axis=0)
            print(model.predict(input_data)[0])

        if not found:
            write(frame, 'no face')
        elif model.predict(input_data)[0] < .6:
            write(frame, ':)', cor=cor, clr=True)
        else:
            write(frame, ':|', cor=cor)


        cv2.imshow('Frame', frame)

        cv2.waitKey(1)


def write(img, msg, clr=False, cor=None):
    font = cv2.FONT_HERSHEY_SIMPLEX;
    font_scale = 1
    thickness = 1
    color = (0, 255, 0) if clr else (0, 0, 255)

    if type(cor) is np.ndarray:
        cv2.putText(img, msg, (cor[0], cor[1]), font, font_scale, color, thickness)
        draw_rectangle(img, coordinate=cor, color=(255, 255, 255))
    else:
        cv2.putText(img, msg, (200, 200), font, font_scale, color, thickness)
