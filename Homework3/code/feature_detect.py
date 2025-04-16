import typing
import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect_features(img: np.ndarray, visualize: bool = False):
    orb = cv2.ORB_create(
        nfeatures=2000,  # max number of features to retain
        scaleFactor=1.2,  # scale factor between levels in the pyramid
        nlevels=8,  # number of levels in the pyramid
        edgeThreshold=31,  # size of the border where features are not detected
        firstLevel=0,  # level of pyramid to start from
        WTA_K=2,  # number of points that produce each feature
        scoreType=cv2.ORB_HARRIS_SCORE,  # Harris score for feature detection
        patchSize=31,  # size of the patch used by the Harris detector
    )

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
