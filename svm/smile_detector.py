import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf

from svm.preprocessor import Preprocessor


class SmileDetector:
    def __init__(self, x, y, model_file):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

        self.x_test = np.array(self.x_test)
        self.x_train = np.array(self.x_train)
        self.y_test = np.array(self.y_test)
        self.x_train = np.array(self.x_train)

        self.model_file = model_file
        self._model = None

    def train_svm(self, svm_kernel):
        # use L2 to avoid overfitting (C=.7)
        self._model = svm.SVC(kernel=svm_kernel, C=.7)
        self._model.fit(self.x_train, self.y_train)

        self.train_accuracy()
        self.save_model()

    def save_model(self):
        pickle.dump(self._model, open(self.model_file, 'wb'))

    def load_model(self):
        self._model = pickle.load(open(self.model_file, "rb"))

    def train_accuracy(self):
        y_pred = self._model.predict(self.x_train)
        num_correct = np.sum(y_pred == self.y_train)
        accuracy = num_correct / len(self.y_train)
        print(f"train Accuracy: {accuracy * 100}%")

    def test_accuracy(self):
        y_pred = self._model.predict(self.x_test)
        num_correct = np.sum(y_pred == self.y_test)
        accuracy = num_correct / len(self.y_test)
        print(f"test Accuracy: {accuracy * 100}%")

    def get_model(self):
        return self._model

    def confusion_matrix(self):
        y_pred = self._model.predict(self.x_test)

        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()