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


class SmileDetector:
    def __init__(self, x, y, model_file):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

        self.x_test = np.array(self.x_test)
        self.x_train = np.array(self.x_train)
        self.y_test = np.array(self.y_test)
        self.x_train = np.array(self.x_train)

        self.input_shape = (*self.x_train[0].shape, 1)
        print(self.input_shape)
        self.model_file = model_file
        self._model = None

    def train(self, epoch):
        self._model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self._model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

        self._model.summary()
        self._model.fit(self.x_train, self.y_train, epochs=epoch)
        self.train_accuracy()
        self.save_model()

    def train_complex(self, epoch):
        base_model = ResNet50(weights=None, include_top=False, input_shape=self.input_shape)
        self._model = Sequential()

        self._model.add(base_model)

        self._model.add(Flatten())
        self._model.add(Dense(256, activation='relu'))
        self._model.add(Dense(1, activation='sigmoid'))

        self._model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

        self._model.summary()
        self._model.fit(self.x_train, self.y_train, epochs=epoch)
        self.train_accuracy()
        self.save_model()

    def save_model(self):
        pickle.dump(self._model, open(self.model_file, 'wb'))

    def load_model(self):
        self._model = pickle.load(open(self.model_file, "rb"))

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
