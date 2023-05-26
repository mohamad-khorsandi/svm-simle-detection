import cv2

def multi_detect_face(img):
    face_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_frontalface_alt2.xml')
    faces = _object_detection(face_cascade, img)
    if len(faces) == 0:
        return None, True

    return faces, False

def detect_face(img):
    face_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_frontalface_alt2.xml')
    coordinates = _object_detection(face_cascade, img)
    if len(coordinates) == 0:
        return None
    # todo if len(faces) > 1
    return coordinates[0]


def detect_mouth(img):
    mouth_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_mcs_mouth.xml')
    coordinates = _object_detection(mouth_cascade, img)

    if len(coordinates) == 0:
        return None
    # todo if len(faces) > 1
    return coordinates[0]


def _object_detection(cascade_classifier: cv2.CascadeClassifier, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
