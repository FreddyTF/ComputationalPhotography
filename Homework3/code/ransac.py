import numpy as np
import cv2
import typing
import random
from homography import apply_homography, invert_homography
from matplotlib import pyplot as plt


def ransac(
    matches,
    img1,
    img2,
    kp1,
    kp2,
    iterations: int = 1000,
    patch_size: int = 1,
    threshold: float = 5.0,
):
    n = iterations  # Number of iterations

    best_inliers = []
    best_H = None

    outliers_count_best = len(matches) + 1

    for x in range(n):
        # Select 4 random matches
        random_matches = random.sample(matches, 4)

        # compute the indexes of the matches
        indexes_1 = np.array([match.queryIdx for match in random_matches])
        indexes_2 = np.array([match.trainIdx for match in random_matches])

        # Get the points from the matches to get the pixels
        points1 = np.array([kp1[i].pt for i in indexes_1], dtype=np.float32)
        points2 = np.array([kp2[i].pt for i in indexes_2], dtype=np.float32)

        # Compute the homography matrix using the selected points
        H, _ = cv2.findHomography(points1, points2)

        # Compute the inliers and outliers
        # reset to empty lists
        inliers = []
        outliers_count = 0
        inliers_count = 0

        for match in matches:
            # make computing faster by skipping the matches that are roven to be worse than the best
            if outliers_count > outliers_count_best:
                # print("Skipping match")
                continue
            # Get the points from the match
            point1 = np.array(kp1[match.queryIdx].pt, dtype=np.int32)
            point2 = np.array(kp2[match.trainIdx].pt, dtype=np.int32)

            homographed_point = apply_homography(H, point1)
            try:
                homographed_point = np.array(
                    [int(homographed_point[0]), int(homographed_point[1])],
                    dtype=np.int32,
                )
            except Exception as e:
                print(f"Error in homography: {e}")
                outliers_count += 1
                continue

            if (
                (0 <= homographed_point[1])
                and (homographed_point[1] < img2.shape[0])
                and (0 <= homographed_point[0])
                and (homographed_point[0] < img2.shape[1])
            ):
                reprojection_error = np.linalg.norm(homographed_point - point2, 2)

                if reprojection_error < threshold:
                    inliers.append(match)
                    inliers_count += 1
                else:
                    outliers_count += 1

        if len(inliers) > len(best_inliers):
            print("Found better inliers")
            print(f"H: {H}")

            best_inliers = inliers.copy()
            best_H = H
            outliers_count_best = outliers_count

    print(f"Number of inliers: {len(best_inliers)}")
    print(f"Number of outliers: {len(matches) - len(best_inliers)}")
    print(f"Homography matrix: {best_H}")
    return best_inliers, best_H
