import pytest
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

from homography import compute_homography_from_matches
import cv2
import glob


def test_compute_homography_from_matches():
    # Create dummy keypoints in image 1 (grid of points)
    pts1 = np.array(
        [[x, y] for x in range(20, 100, 20) for y in range(20, 100, 20)],
        dtype=np.float32,
    )
    pts1 = pts1.reshape(-1, 1, 2)  # shape: (N, 1, 2)

    # Rotation matrix (45 degrees) + translation (x=1, y=10)
    angle = np.deg2rad(45)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    H = np.array([[cos_a, -sin_a, 1], [sin_a, cos_a, 10], [0, 0, 1]], dtype=np.float32)

    # Apply homography to generate points in image 2
    pts2 = cv2.perspectiveTransform(pts1, H)

    # Create KeyPoint objects
    kp1 = [cv2.KeyPoint(x=float(p[0][0]), y=float(p[0][1]), _size=1) for p in pts1]
    kp2 = [cv2.KeyPoint(x=float(p[0][0]), y=float(p[0][1]), _size=1) for p in pts2]

    # Create dummy matches (1-to-1 matching)
    matches = [
        cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(len(kp1))
    ]

    # Extract points from matches
    src_pts = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    # Call the function
    computed_homography, _ = cv2.findHomography(src_pts, dst_pts, method=0) # 0 for least squares

    # Assert the result
    assert computed_homography is not None
    assert np.allclose(computed_homography, H, atol=1e-1)
