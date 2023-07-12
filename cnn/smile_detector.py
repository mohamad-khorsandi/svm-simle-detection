import os.path
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from .preprocessor import Preprocessor
from keras.applications.vgg16 import VGG16
from keras.applications import EfficientNetV2B0
from keras.applications import MobileNet, ResNet50
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

model_path = 'models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'

class SmileDetector:
    def __init__(self, x, y, model_file):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

        self.x_test = np.array(self.x_test)
        self.x_train = np.array(self.x_train)
        self.y_test = np.array(self.y_test)
        self.x_train = np.array(self.x_train)

        self.input_shape = x[0].shape
        print(self.input_shape)
        self.model_file = f'{model_file}.h5'
        self._model = None

    def train(self, epoch):
        base_model = tf.keras.applications.MobileNetV2(input_shape=self.input_shape,
                                                       include_top=False,
                                                       weights=None)
        base_model.load_weights(model_path)

        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.Sequential([
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        base_model.trainable = True

        inputs = tf.keras.Input(shape=self.input_shape)
        x = preprocess_input(inputs)
        x = base_model(x, training=True)
        x = global_average_layer(x)
        # x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        self._model = tf.keras.Model(inputs, outputs)

        self._model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

        self._model.summary()
        self._model.fit(self.x_train, self.y_train, epochs=epoch)
        self.train_accuracy()
        self.save_model()

    def fit_more(self, epoch):
        self.load_model()
        self._model.fit(self.x_train, self.y_train, epochs=epoch)

    def save_model(self):
        print(self.model_file)
        self._model.save(self.model_file)

    def load_model(self):
        print(self.model_file)
        self._model = tf.keras.models.load_model(self.model_file)

    def train_accuracy(self):
        accuracy = self._model.evaluate(self.x_train, self.y_train)[1]
        print(f"train Accuracy: {accuracy * 100}%")

    def test_accuracy(self):
        accuracy = self._model.evaluate(self.x_test, self.y_test)[1]
        print(f"test Accuracy: {accuracy * 100}%")

    def confusion_matrix(self):
        y_pred = self.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def predict(self, x):
        pred = []
        threshold = 0.5
        y_pred_prob = self._model.predict(self.x_test)
        for prob in y_pred_prob:
            if prob[0] > threshold:
                pred.append(1)
            else:
                pred.append(0)
        return pred

    def single_predict(self, face):
        return self._model.predict(face)
