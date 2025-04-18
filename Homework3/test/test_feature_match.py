import pytest
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

from feature_detect import detect_features
from feature_match import match_features
import cv2
import glob


def test_feature_detection():
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

    matches = match_features(img1, kp1, des1, img2, kp2, des2, visualize=True)

    assert matches is not None, "Matches should not be None"
    assert len(matches) > 0, "Matches should be found"
    
