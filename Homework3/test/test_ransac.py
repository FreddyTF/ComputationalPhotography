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
from matplotlib import pyplot as plt


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


def test_compare_ransac():
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

    ransac_matches_sift, _ = ransac(
        match_features_sift,
        img1,
        img2,
        kp1_sift,
        kp2_sift,
        threshold=2.5,
        iterations=3000,
    )

    ransac_matches_orb, _ = ransac(
        match_features_orb,
        img1,
        img2,
        kp1_orb,
        kp2_orb,
        threshold=2.5,
        iterations=3000,
    )

    img_match_sift = cv2.drawMatches(
        img1,
        kp1_sift,
        img2,
        kp2_sift,
        ransac_matches_sift,
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
        ransac_matches_orb,
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
    axes[0].set_title(f"RANSAC ORB Matches: {len(ransac_matches_orb)}")
    axes[0].axis("off")

    # SIFT features
    axes[1].imshow(img_match_sift)
    axes[1].set_title(f"RANSAC SIFT Matches: {len(ransac_matches_sift)}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
