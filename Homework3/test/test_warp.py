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
from warp import warp_images_inverse
from homography import compute_average_homography


def test_warp():
    # Test if the feature detection function works correctly
    image_files = glob.glob(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../images/panorama1/*.jpeg")
        )
    )
    # in this case always 2 files

    img1 = cv2.imread(image_files[1])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    kp1, des1 = detect_features(img1, visualize=False)

    img2 = cv2.imread(image_files[0])
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    kp2, des2 = detect_features(img2, visualize=False)

    matches = match_features(img1, kp1, des1, img2, kp2, des2, visualize=False)

    ransac_matches, H = ransac(
        matches, img1, img2, kp1, kp2, threshold=3.0, patch_size=7, iterations=1000
    )

    homography = compute_average_homography(
        ransac_matches, kp1, kp2, img1, img2, visualize=True
    )

    print(f"Homography: {homography}")
    assert homography is not None, "Homography should not be None"

    result = warp_images_inverse(H, img1, img2, visualize=True)

    assert result is not None, "Warped image should not be None"


def test_warp_2():
    # Test if the feature detection function works correctly
    image_files = glob.glob(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../images/panorama4/*.jpeg")
        )
    )
    # in this case always 2 files

    img1 = cv2.imread(image_files[1])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    kp1, des1 = detect_features(img1, visualize=False)

    img2 = cv2.imread(image_files[2])
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    kp2, des2 = detect_features(img2, visualize=False)

    matches = match_features(img1, kp1, des1, img2, kp2, des2, visualize=True)

    ransac_matches, H = ransac(
        matches, img1, img2, kp1, kp2, threshold=2.0, patch_size=7, iterations=2000
    )

    homography = compute_average_homography(
        ransac_matches, kp1, kp2, img1, img2, visualize=True
    )

    print(f"Homography: {homography}")
    assert homography is not None, "Homography should not be None"

    result = warp_images_inverse(homography, img1, img2, visualize=False)

    assert result is not None, "Warped image should not be None"
