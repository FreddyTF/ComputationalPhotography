import cv2
import numpy as np
import typing


def compute_homography_from_matches(matches):
    computed_homography = cv2.findHomography(
        np.array([match.queryIdx for match in matches]),
        np.array([match.trainIdx for match in matches]),
        cv2.RANSAC,
    )

    return computed_homography


def invert_homography(homography):
    # Invert the homography matrix
    return np.linalg.inv(homography)
