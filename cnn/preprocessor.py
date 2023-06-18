import cv2
import numpy as np

from utils import image_utils
from utils.image_utils import crop


class Preprocessor:
    def __init__(self, images, single_image=False, images_shape=(64, 64)):
        self.single_image = single_image
        self.single_image_cord = None
        self.images = []
        if single_image:
            self.images.append(images)
        else:
            self.images = images

        self.face_images = []
        self.extract_faces()
        self.to_gray()
        self.resize_face_images(images_shape)

    def extract_faces(self):
        for image in self.images:
            cor, found = image_utils.detect_face(image)
            if not found:
                self.face_images.append(image)
            else:
                if self.single_image:
                    self.single_image_cord = cor
                self.face_images.append(crop(image, cor))

    def resize_face_images(self, new_size):
        resized_images = []
        for image in self.face_images:
            resized_image = cv2.resize(image, new_size)
            resized_images.append(resized_image)

        self.face_images = resized_images

    def get_x(self):
        if self.single_image:
            return np.array(self.face_images), self.single_image_cord
        else:
            return np.array(self.face_images)

    def to_gray(self):
        gray_images = []
        for image in self.face_images:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_images.append(gray_img)

        self.face_images = gray_images
