import cv2
import numpy as np


class Preprocessor:
    def __init__(self, images, images_shape=(64, 64)):
        self.face_images = images

        self.resize_face_images(images_shape)

    def resize_face_images(self, new_size):
        resized_images = []
        for image in self.face_images:
            resized_image = cv2.resize(image, new_size)
            resized_images.append(resized_image)

        self.face_images = resized_images

    def get_x(self):
        return np.array(self.face_images)

    # in case you want to use gray images
    def to_gray(self):
        gray_images = []
        for image in self.face_images:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_images.append(gray_img)

        self.face_images = gray_images
