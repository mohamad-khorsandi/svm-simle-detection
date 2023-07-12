import os.path
import pickle

import video_streaming
from svm.smile_detector import SmileDetector as SvmSmileDetector
from svm.preprocessor import Preprocessor as SvmPreprocessor

from cnn.smile_detector import SmileDetector as CnnSmileDetector
from cnn.preprocessor import Preprocessor as CnnPreprocessor
import tensorflow as tf

from utils import file_utils, image_utils

model_svm = os.path.join('models', 'model')
model_cnn = os.path.join('models', 'cnn_model_cur')


def live_stream():
    model = tf.keras.models.load_model('models/cnn_model_cur.h5')

    video_streaming.camera_smile_detection(model)


def train_and_store_svm():
    images = file_utils.load_images()
    y = file_utils.load_labels()
    preprocessor = SvmPreprocessor(images)

    smile_detector = SvmSmileDetector(preprocessor.get_x(), y, model_cnn)
    smile_detector.train_svm('rbf')
    smile_detector.test_accuracy()
    smile_detector.confusion_matrix()
    return smile_detector


def train_and_store_cnn():
    images = file_utils.load_faces()
    y = file_utils.load_labels()
    preprocessor = CnnPreprocessor(images, images_shape=(64, 64))
    smile_detector = CnnSmileDetector(preprocessor.get_x(), y, model_cnn)
    smile_detector.train(3)
    smile_detector.test_accuracy()
    smile_detector.confusion_matrix()

    return smile_detector


if __name__ == '__main__':
    live_stream()
