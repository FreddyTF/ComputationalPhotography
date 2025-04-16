import numpy as np
import cv2
import typing


def ransac(matches, iterations: int = 1000, threshold: float = 5.0):
    n = 1000  # Number of iterations

    best_inliers = []

    for x in range(n):
        # Select 4 random matches
        random_matches = np.random.choice(matches, 4, replace=False)

        # Extract the points from the matches
        points1 = np.array([match.queryIdx for match in random_matches])
        points2 = np.array([match.trainIdx for match in random_matches])

        # Compute the homography matrix using the selected points
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Compute the inliers and outliers
        inliers = []
        outliers = []
        threshold = 5.0  # Define a threshold for inliers

        for match in matches:
            point1 = np.array([match.queryIdx])
            point2 = np.array([match.trainIdx])

            # Compute the reprojection error
            reprojection_error = np.abs(
                np.dot(H, point1) - point2
            )  # TODO: use SSD instead of L2 norm

            if reprojection_error < threshold:
                inliers.append(match)
            else:
                outliers.append(match)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    return best_inliers



