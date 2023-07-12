import cv2
from matplotlib import pyplot as plt
from skimage import io


def show_gray_img(image):
    io.imshow(image, cmap='gray')
    io.show()


def show_cv2_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    io.imshow(image)
    io.show()


def show_img(image):
    io.imshow(image)
    io.show()


def crop(img, coordinate):
    if coordinate is None:
        return img

    (x, y, w, h) = coordinate
    return img[x:x + w, y: y + h]


def draw_rectangle(image, coordinate, color):
    res = image
    (x, y, w, h) = coordinate
    cv2.rectangle(res, (x, y), (x + w, y + h), color, 1)
    return res


def detect_face(img, multi=False):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None, False

    if multi:
        return faces, True
    else:
        return faces[0], True
