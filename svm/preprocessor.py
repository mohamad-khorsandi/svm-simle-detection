import cv2
import numpy as np
import skimage
from skimage.feature import hog
from skimage import exposure, feature
from utils import image_utils
from utils.image_utils import crop


class Preprocessor:
    def __init__(self, images, single_image=False, images_shape=(64, 64)):
        self.single_image = single_image
        self.images = []
        if single_image:
            self.images.append(images)
        else:
            self.images = images

        self.face_images = []
        self.extract_faces()
        self.resize_face_images(images_shape)

        self._x = None
        self.extract_features()
        self.normalize()

    def extract_faces(self):
        for image in self.images:
            cor, found = image_utils.detect_face(image)
            if not found:
                self.face_images.append(image)
            else:
                self.face_images.append(crop(image, cor))

    def resize_face_images(self, new_size):
        resized_images = []
        for image in self.face_images:
            resized_image = cv2.resize(image, new_size)
            resized_images.append(resized_image)

        self.face_images = resized_images

    def extract_features(self):
        feature_list = []
        for i, face in enumerate(self.face_images):
            lbp_features = self.extract_lbp(face)
            hog_features = self.extract_hog(face)
            features = np.concatenate((lbp_features, hog_features))
            feature_list.append(features)

        self._x = np.array(feature_list)

    def normalize(self):
        min_vals = self._x.min(axis=0)
        max_vals = self._x.max(axis=0)
        self._x = (self._x - min_vals) / (max_vals - min_vals)

    def get_x(self):
        return self._x

    @classmethod
    def extract_hog(cls, image):
        fd = hog(image, channel_axis=-1)
        return fd

    @classmethod
    def extract_lbp(cls, image, radius=3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, 8*radius, radius, method='uniform')
        return lbp.astype(np.uint8).ravel()

