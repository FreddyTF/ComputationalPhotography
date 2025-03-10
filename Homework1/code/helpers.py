import cv2
import numpy as np
import typing
from typing import List
import os


def readImagesToNumpyArray(dir_path: str) -> List[np.ndarray]:
    """
    Read all images in a directory to a numpy array.
    :param dir_path: The path to the directory containing the images
    :return: A numpy array containing all the images
    """
    images = []

    for filename in os.listdir(dir_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(dir_path, filename))
            if img is not None:
                images.append(img)

    return images
