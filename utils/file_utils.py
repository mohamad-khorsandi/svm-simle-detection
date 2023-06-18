import os
import random
import shutil
from os import listdir

import cv2
import numpy as np

images_dir = os.path.join('dataset', 'files')
labels_file = os.path.join('dataset', 'labels.txt')


def load_images(s=0, e=4000):
    dataset = []
    all_images = sorted(listdir(images_dir))[s:e]
    for filename in all_images:
        dataset.append(cv2.imread(os.path.join(images_dir, filename)))

    return dataset


def load_labels(s=0, e=4000):
    with open(labels_file, 'r') as file:
        labels = list()
        for line in file:
            labels.append(int(line.split(' ')[0]))

        print("smile percent: ", np.count_nonzero(np.array(labels[s:e])) / (e - s) * 100, "%")
        return np.array(labels[s:e])
