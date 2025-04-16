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

def apply_homography(homography, point):
    # Apply the homography to a point
    point_homogeneous = np.array([point[0], point[1], 1])
    transformed_point = np.dot(homography, point_homogeneous)
    transformed_point /= transformed_point[2]  # Normalize
    return transformed_point[:2]  # Return only x and y coordinates

def invert_homography(homography):
    # Invert the homography matrix
    return np.linalg.inv(homography)
