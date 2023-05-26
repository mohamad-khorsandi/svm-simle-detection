import pickle

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from utils import file_utils, image_utils, object_detection
from utils.file_utils import faces_dir, dataset_dir, load_images
from feature_extractors import FeatureExtractors
from utils.image_utils import crop


class SmileDetector:
    def __init__(self, model_file):
        self.model_file = model_file
        self.model = None
        self.face_images = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def preprocess(self):
        print('load')
        images = load_images(dataset_dir)
        print('fx')
        self.face_images = self.extract_faces(images)
        features = self.extract_features()
        labels = file_utils.load_labels()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.2, shuffle=True)

    def train(self, svm_kernel):
        self.model = svm.SVC(kernel=svm_kernel)
        self.model.fit(self.x_train, self.y_train)
        accuracy = self.model.score(self.x_train, self.y_train)
        print("train Accuracy:", accuracy)

    def extract_faces(self, images):
        faces = []
        for image in images:
            cor = object_detection.detect_face(image)
            faces.append(crop(image, cor))
        return faces

    def extract_features(self):
        feature_list = []
        for i, face in enumerate(self.face_images):
            print(i)
            feature_list.append(self.face_features(face))

        return np.vstack(feature_list)

    def face_features(self, face):
        hog = FeatureExtractors.extract_hog(face)
        lbp = FeatureExtractors.extract_lbp(face)
        return np.concatenate((hog, lbp))


    def save_model(self):
        assert self.model is not None
        pickle.dump(self.model, open(self.model_file, 'wb'))

    def load_model(self):
        self.model = pickle.load(open(self.model_file, "rb"))

    def accuracy(self):
        accuracy = self.model.score(self.x_test, self.y_test)
        print("test Accuracy:", accuracy)

    def predict(self, face):
        features = self.face_features(face)
        return self.model.predict(features.reshape(1,-1))[0]