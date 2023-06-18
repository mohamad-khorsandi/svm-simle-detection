import os.path
import pickle

import video_streaming
from svm.smile_detector import SmileDetector as SvmSmileDetector
from svm.preprocessor import Preprocessor as SvmPreprocessor

from cnn.smile_detector import SmileDetector as CnnSmileDetector
from cnn.preprocessor import Preprocessor as CnnPreprocessor


from utils import file_utils, image_utils
import cv2

model_svm = os.path.join('models', 'model')
model_cnn = os.path.join('models', 'cnn_model_cur')


def live_stream():
    model = pickle.load(open(model_cnn, "rb"))
    video_streaming.camera_smile_detection(model)


def train_and_store_svm():
    images = file_utils.load_images(1900, 2100)
    y = file_utils.load_labels(1900, 2100)
    preprocessor = SvmPreprocessor(images)

    smile_detector = SvmSmileDetector(preprocessor.get_x(), y, model_cnn)
    smile_detector.train_svm('rbf')
    smile_detector.test_accuracy()
    smile_detector.confusion_matrix()
    return smile_detector


def train_and_store_cnn():
    images = file_utils.load_images()
    y = file_utils.load_labels()
    preprocessor = CnnPreprocessor(images, images_shape=(40, 40))
    smile_detector = CnnSmileDetector(preprocessor.get_x(), y, model_cnn)
    smile_detector.train(8)
    smile_detector.test_accuracy()
    smile_detector.confusion_matrix()

    return smile_detector


if __name__ == '__main__':
    train_and_store_cnn()
