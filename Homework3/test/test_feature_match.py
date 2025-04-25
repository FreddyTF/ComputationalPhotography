import pytest
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

from feature_detect import detect_features
from feature_match import match_features
import cv2
import glob
from matplotlib import pyplot as plt


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


def test_compare_feature_detection_sift_orb():
    # Test if ORB and SIFT feature detection methods work correctly
    image_files = glob.glob(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../images/panorama5/*.jpeg")
        )
    )
    img1 = cv2.imread(image_files[0])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img2 = cv2.imread(image_files[1])
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    kp1_orb, des1_orb = detect_features(img1, mode="orb")
    kp2_orb, des2_orb = detect_features(img2, mode="orb")

    match_features_orb = match_features(
        img1,
        kp1_orb,
        des1_orb,
        img2,
        kp2_orb,
        des2_orb,
        features="orb",
        visualize=False,
    )

    kp1_sift, des1_sift = detect_features(img1, mode="sift")
    kp2_sift, des2_sift = detect_features(img2, mode="sift")

    match_features_sift = match_features(
        img1,
        kp1_sift,
        des1_sift,
        img2,
        kp2_sift,
        des2_sift,
        features="sift",
        visualize=False,
    )

    img_match_sift = cv2.drawMatches(
        img1,
        kp1_sift,
        img2,
        kp2_sift,
        match_features_sift[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 0, 255),  # Green color for matches
        singlePointColor=None,
        matchesThickness=10,  # Increase the thickness of the match lines
    )

    img_match_orb = cv2.drawMatches(
        img1,
        kp1_orb,
        img2,
        kp2_orb,
        match_features_orb[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 0, 255),  # Green color for matches
        singlePointColor=None,
        matchesThickness=10,  # Increase the thickness of the match lines
    )

    # Create a plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # ORB features
    axes[0].imshow(img_match_orb)
    axes[0].set_title(f"ORB Matches: {len(match_features_orb)}")
    axes[0].axis("off")

    # SIFT features
    axes[1].imshow(img_match_sift)
    axes[1].set_title(f"SIFT Matches: {len(match_features_sift)}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
