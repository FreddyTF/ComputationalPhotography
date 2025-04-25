import pytest
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

from feature_detect import detect_features
import cv2
import glob
import matplotlib.pyplot as plt


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


def test_feature_detection_orb_vs_sift():
    # Test if ORB and SIFT feature detection methods work correctly
    image_files = glob.glob(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../images/panorama5/*.jpeg")
        )
    )
    img = cv2.imread(image_files[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kp_orb, des_orb = detect_features(img, mode="orb")
    kp_sift, des_sift = detect_features(img, mode="sift")

    # Limit the number of features to 50 for visualization
    # kp_orb_2 = kp_orb[:50]
    # kp_sift_2 = kp_sift[:50]

    # Draw keypoints on the images
    img_orb = cv2.drawKeypoints(
        img,
        kp_orb,
        None,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    img_sift = cv2.drawKeypoints(
        img,
        kp_sift,
        None,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # Create a plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ORB features
    axes[0].imshow(img_orb)
    axes[0].set_title(f"ORB Features: {len(kp_orb)}")
    axes[0].axis("off")

    # SIFT features
    axes[1].imshow(img_sift)
    axes[1].set_title(f"SIFT Features: {len(kp_sift)}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
