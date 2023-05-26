import cv2
from smile_detection import SmileDetector
from utils.image_utils import crop, draw_rectangle
from utils.object_detection import detect_face, multi_detect_face


def camera_smile_detection(sd: SmileDetector):
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        coordinates, no_face = multi_detect_face(frame)
        if no_face:
            continue

        coordinates = coordinates[0]

        face = crop(frame, coordinates)

        if sd.predict(face) == 1:
            draw_rectangle(frame, coordinates, (0,255,0))
        else:
            draw_rectangle(frame, coordinates, (0,0,255))
        cv2.imshow('Frame', frame)

        cv2.waitKey(1)

