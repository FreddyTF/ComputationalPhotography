import cv2
import numpy as np
import typing
import matplotlib.pyplot as plt
import time


def apply_homography(homography, point):
    # Apply the homography to a point
    point_homogeneous = np.array([point[0], point[1], 1])
    transformed_point = np.dot(homography, point_homogeneous)
    transformed_point /= transformed_point[2]  # Normalize
    return transformed_point[:2]  # Return only x and y coordinates

def invert_homography(homography):
    # Invert the homography matrix
    return np.linalg.inv(homography)

def compute_average_homography(best_inliers, kp1, kp2, img1, img2, visualize=False):
    """
    Compute the average homography matrix using the best inliers.
    """
    homography, _ = cv2.findHomography(
        np.array([kp1[match.queryIdx].pt for match in best_inliers], dtype=np.float32),
        np.array([kp2[match.trainIdx].pt for match in best_inliers], dtype=np.float32),
        method=0,  # 0 for least squares
    )

    if visualize:
        img3 = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            best_inliers,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 0, 255),  # Green color for matches
            singlePointColor=None,
            matchesThickness=10,  # Increase the thickness of the match lines
        )
        plt.imshow(img3)
        plt.savefig(
            r"report\images\8\_" + time.strftime("%Y-%m-%d_%H-%M-%S") + ".png",
            dpi=300,  # high resolution
        )
        plt.close()

    return homography
