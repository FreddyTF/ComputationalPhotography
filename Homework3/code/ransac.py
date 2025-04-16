import numpy as np
import cv2
import typing
import random
from homography import apply_homography, invert_homography
from matplotlib import pyplot as plt


def compute_ssd_of_neighborhood(img1, img2, point1, point2):
    points1 = img1[point1[1] - 1 : point1[1] + 2, point1[0] - 1 : point1[0] + 2]
    points2 = img2[point2[1] - 1 : point2[1] + 2, point2[0] - 1 : point2[0] + 2]

    # Compute the sum of squared differences (SSD)
    ssd = np.sum((points1 - points2) ** 2)

    return ssd


def ransac(
    matches,
    img1,
    img2,
    kp1,
    kp2,
    iterations: int = 1000,
    threshold: float = 5.0,
):
    n = 100  # Number of iterations

    best_inliers = []

    for x in range(n):
        # Select 4 random matches

        random_matches = random.sample(matches, 4)

        if False:
            img3 = cv2.drawMatches(
                img1,
                kp1,
                img2,
                kp2,
                random_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                matchColor=(0, 0, 255),  # Green color for matches
                singlePointColor=None,
                matchesThickness=10,  # Increase the thickness of the match lines
            )

            plt.imshow(img3), plt.show()

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
        threshold =  2000.0  # Define a threshold for inliers

        for match in matches:
            # Get the points from the match
            point1 = np.array(kp1[match.queryIdx].pt, dtype=np.float32)
            img1_pxl = np.array([point1[0], point1[1]])
            point2 = np.array(kp2[match.trainIdx].pt, dtype=np.float32)

            point2 = np.array([int(point2[0]), int(point2[1])], dtype=np.int32)

            homographed_point = apply_homography(H, img1_pxl)
            homographed_point = np.array(
                [int(homographed_point[0]), int(homographed_point[1])], dtype=np.int32
            )
            if (
                (1 < homographed_point[1])
                and (homographed_point[1] < img1.shape[0] - 1)
                and (1 < homographed_point[0])
                and (homographed_point[0] < img1.shape[1] - 1)
            ):
                reprojection_error = compute_ssd_of_neighborhood(
                    img1, img2, homographed_point, point2
                )

                if reprojection_error < threshold:
                    inliers.append(match)
                else:
                    outliers.append(match)
            else:
                outliers.append(match)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers.copy()

    print(f"Number of inliers: {len(best_inliers)}")
    print(f"Number of outliers: {len(matches) - len(best_inliers)}")
    print(f"Homography matrix: {H}")
    return best_inliers
