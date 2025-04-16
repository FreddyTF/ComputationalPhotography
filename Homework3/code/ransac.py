import numpy as np
import cv2
import typing
import random
from homography import apply_homography, invert_homography


def ransac(
    matches,
    img1,
    img2,
    kp1,
    kp2,
    iterations: int = 1000,
    threshold: float = 5.0,
):
    n = 200  # Number of iterations

    best_inliers = []

    for x in range(n):
        # Select 4 random matches

        random_matches = random.sample(matches, 4)

        # Extract the points from the matches
        # points1 = np.array([match.queryIdx for match in random_matches])
        # points2 = np.array([match.trainIdx for match in random_matches])

        indexes_1 = np.array([match.queryIdx for match in random_matches])
        indexes_2 = np.array([match.trainIdx for match in random_matches])

        points1 = np.array([kp1[i].pt for i in indexes_1], dtype=np.float32)
        points2 = np.array([kp2[i].pt for i in indexes_2], dtype=np.float32)

        # Compute the homography matrix using the selected points
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Compute the inliers and outliers
        inliers = []
        outliers = []
        threshold = 100.0  # Define a threshold for inliers

        for match in matches:
            # Get the points from the match
            point1 = np.array(kp1[match.queryIdx].pt, dtype=np.float32)
            img1_pxl = np.array([point1[0], point1[1]])
            point2 = np.array(kp2[match.trainIdx].pt, dtype=np.float32)
            img2_pxl = np.array([point2[0], point2[1]])
            #H = invert_homography(H)
            homographed_point = apply_homography(H, img2_pxl)

            img1_color = img1[int(img1_pxl[1]), int(img1_pxl[0])]
            if (
                (0 < homographed_point[0])
                and (homographed_point[0] < img2.shape[1])
                and (0 < homographed_point[1])
                and (homographed_point[1] < img2.shape[0])
            ):
                img2_color = img2[int(homographed_point[1]), int(homographed_point[0])]

                reprojection_error = np.sum(img1_color - img2_color) ** 2 

                if reprojection_error < threshold:
                    inliers.append(match)
                else:
                    outliers.append(match)
            else:
                outliers.append(match)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    print(f"Number of inliers: {len(best_inliers)}")
    print(f"Number of outliers: {len(matches) - len(best_inliers)}")
    print(f"Homography matrix: {H}")
    return best_inliers
