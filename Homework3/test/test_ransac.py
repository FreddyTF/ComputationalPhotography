import pytest
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

from feature_detect import detect_features
from feature_match import match_features
from ransac import ransac, compute_ssd_of_neighborhood
import cv2
import glob


def test_ransac():
    # Test if the feature detection function works correctly
    image_files = glob.glob(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../images/panorama1/*.jpeg")
        )
    )
    # in this case always 2 files

    img1 = cv2.imread(image_files[0])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    kp1, des1 = detect_features(img1, visualize=False)

    img2 = cv2.imread(image_files[1])
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    kp2, des2 = detect_features(img2, visualize=False)

    matches = match_features(img1, kp1, des1, img2, kp2, des2, visualize=False)

    ransac_matches = ransac(
        matches, img1, img2, kp1, kp2, threshold=5.0, iterations=10000
    )

    assert ransac_matches is not None, "RANSAC matches should not be None"


def test_compute_ssd_of_neighborhood():
    # Create two synthetic images
    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
    img2 = np.ones((10, 10, 3), dtype=np.uint8)

    # Add a small patch in both images
    img1[3:8, 3:8] = 100
    img2[3:8, 3:8] = 100

    # Define points at the center of the patches
    point1 = (5, 5)
    point2 = (5, 5)

    # Compute SSD for identical patches
    ssd = compute_ssd_of_neighborhood(img1, img2, point1, point2)
    assert ssd == 0, f"Expected SSD to be 0, but got {ssd}"

    # Modify the second image patch
    img2[3:8, 3:8] = 50

    # Compute SSD for different patches
    ssd = compute_ssd_of_neighborhood(img1, img2, point1, point2, print_debug=True)

    expected_ssd = np.sum((img1[3:8, 3:8] - img2[3:8, 3:8]) ** 2)

    assert ssd == expected_ssd, f"Expected SSD to be {expected_ssd}, but got {ssd}"

   