import typing
import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect_features(img: np.ndarray, mode: str = "orb", visualize: bool = False):
    if mode == "orb":
        orb = cv2.ORB_create()

        # compute the descriptors with ORB
        kp, des = orb.detectAndCompute(img, None)

        if visualize:
            # draw only keypoints location,not size and orientation
            img2 = cv2.drawKeypoints(
                img,
                kp,
                None,
                color=(0, 0, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            plt.imshow(img2), plt.show()

        return kp, des
    elif mode == "sift":
        sift = cv2.SIFT_create(
            nfeatures=3000,  # max number of features to retain
        )
        kp, des = sift.detectAndCompute(img, None)

        if visualize:
            img2 = cv2.drawKeypoints(
                img,
                kp,
                None,
                color=(0, 0, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            plt.imshow(img2), plt.show()

        return kp, des
