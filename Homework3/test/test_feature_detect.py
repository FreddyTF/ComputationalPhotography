import pytest
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

from feature_detect import detect_features
import cv2
import glob


def test_feature_detection():
    # Test if the feature detection function works correctly
    image_files = glob.glob(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../images/panorama1/*.jpeg")
        )
    )
    for file in image_files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            kp, des = detect_features(img, visualize=True)
            assert kp is not None, "Keypoints should not be None"
            assert des is not None, "Descriptors should not be None"
            assert len(kp) > 0, "Keypoints should be detected"
