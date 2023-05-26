import cv2
import numpy as np
import skimage
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
from sklearn.preprocessing import PowerTransformer

from utils.image_utils import show_gray_img


class FeatureExtractors:
    # used to capture the shape and texture information of objects.
    @classmethod
    def extract_hog(cls, image, show=False):
        image = skimage.transform.resize(image, (64, 64))
        fd, hog_image = hog(image, orientations=16, pixels_per_cell=(16, 16),
                            visualize=True, channel_axis=-1)

        if show:
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            show_gray_img(hog_image_rescaled)

        return fd

    @classmethod
    def extract_lbp(cls, image, show=False, radius=3):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        n_points = 8 * radius

        lbp = local_binary_pattern(gray_image, n_points, 3)

        hist, _ = np.histogram(lbp, bins=range(0, n_points + 3), range=(0, n_points + 2))

        if show:
            picture = (lbp / np.max(lbp) * 255).astype(np.uint8)
            show_gray_img(picture)
        return hist
