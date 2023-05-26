import os
import random
import shutil
from os import listdir

import cv2
import numpy as np

from utils import object_detection
from utils.image_utils import crop

dataset_dir = os.path.join('dataset', 'files')
labels_file = os.path.join('dataset', 'labels.txt')

extracted_data_dir = 'extracted_data'
faces_dir = os.path.join(extracted_data_dir, 'faces')
mouths_dir = os.path.join(extracted_data_dir, 'mouths')

s = 0
e = 4000


def load_images(base_dir):
    dataset = []
    for filename in sorted(listdir(base_dir))[s:e]:
        dataset.append(cv2.imread(os.path.join(base_dir, filename)))

    return dataset


def store_images(images, sub_dir):
    tar_dir = os.path.join(extracted_data_dir, sub_dir)
    if os.path.exists(tar_dir):
        shutil.rmtree(tar_dir)
    os.mkdir(tar_dir)

    for i in range(len(images)):
        filename = os.path.join(tar_dir, f'{i}.jpeg')
        cv2.imwrite(filename, images[i])


def store_faces():
    images = load_images(dataset_dir)
    faces = []
    for image in images:
        cor = object_detection.detect_face(image)
        faces.append(crop(image, cor))
    store_images(faces, faces_dir)


def store_mouths():
    images = load_images(dataset_dir)
    mouths = []
    for image in images:
        cor = object_detection.detect_mouth(image)
        mouths.append(crop(image, cor))
    store_images(mouths, mouths_dir)


def load_rand_image():
    all_images = listdir(dataset_dir)
    img_number = random.randint(2000, len(all_images))
    return cv2.imread(os.path.join(dataset_dir, listdir(dataset_dir)[img_number]))


def load_labels():
    with open(labels_file, 'r') as file:
        labels = list()
        for line in file:
            labels.append(int(line.split(' ')[0]))

        np.count_nonzero(np.array(labels[s:e]))
        return np.array(labels[s:e])