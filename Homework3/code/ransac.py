import numpy as np
import cv2
import typing
import random
from homography import apply_homography, invert_homography
from matplotlib import pyplot as plt


def compute_ssd_of_neighborhood(img1, img2, point1, point2, print_debug=False):
    """
    COmpute the sum of squared differences (SSD) between two patches in two images.
    The patches are centered around the given points.
    The patches are 5x5 pixels in size.
    """
    if (
        point1[1] - 2 < 0
        or point1[1] + 3 > img1.shape[0]
        or point1[0] - 2 < 0
        or point1[0] + 3 > img1.shape[1]
        or point2[1] - 2 < 0
        or point2[1] + 3 > img2.shape[0]
        or point2[0] - 2 < 0
        or point2[0] + 3 > img2.shape[1]
    ):
        return np.inf
        # raise ValueError("Patch goes out of image bounds.")

    points1 = img1[point1[1] - 2 : point1[1] + 3, point1[0] - 2 : point1[0] + 3]
    points2 = img2[point2[1] - 2 : point2[1] + 3, point2[0] - 2 : point2[0] + 3]

    if print_debug:
        print(f"points1: {points1.shape}")
        print(f"points2: {points2.shape}")

    # Compute the sum of squared differences (SSD)
    ssd = np.sum((points1 - points2) ** 2) / (5 * 5 * 3)

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
    n = iterations  # Number of iterations

    best_inliers = []
    best_H = None

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

            plt.axis("off")

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
        outliers = []
        threshold = 100.0  # Define a threshold for inliers
        # H = invert_homography(H)

        for match in matches:
            # Get the points from the match
            point1 = np.array(kp1[match.queryIdx].pt, dtype=np.int32)
            point2 = np.array(kp2[match.trainIdx].pt, dtype=np.int32)

            homographed_point = apply_homography(H, point1)
            homographed_point = np.array(
                [int(homographed_point[0]), int(homographed_point[1])], dtype=np.int32
            )
            if (
                (1 < homographed_point[1])
                and (homographed_point[1] < img1.shape[0] - 1)
                and (1 < homographed_point[0])
                and (homographed_point[0] < img1.shape[1] - 1)
            ):
                if False:
                    # Visualize the homographed point on img1 and point2 on img2
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(img1)
                    axes[0].scatter(
                        homographed_point[0], homographed_point[1], c="b", s=40
                    )
                    axes[0].set_title("Image 1 with Homographed Point")
                    axes[0].axis("off")

                    axes[1].imshow(img2)
                    axes[1].scatter(point2[0], point2[1], c="b", s=40)
                    axes[1].set_title("Image 2 with Point 2")
                    axes[1].axis("off")

                    plt.show()
                    plt.close()

                reprojection_error = compute_ssd_of_neighborhood(
                    img1, img2, point1, homographed_point
                )

                if reprojection_error < threshold:
                    inliers.append(match)

        if len(inliers) > len(best_inliers):
            print("Found better inliers")
            print(f"H: {H}")
            
            best_inliers = inliers.copy()
            best_H = H

    print(f"Number of inliers: {len(best_inliers)}")
    print(f"Number of outliers: {len(matches) - len(best_inliers)}")
    print(f"Homography matrix: {best_H}")
    return best_inliers
