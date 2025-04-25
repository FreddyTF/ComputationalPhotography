import typing
import numpy as np
import cv2
import matplotlib.pyplot as plt


def match_features(
    img1, kp1, des1, img2, kp2, des2, features="sift", visualize: bool = False
):
    if features == "sift":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[: int(len(matches) * 0.25)]
        if visualize:
            # Draw first 10 matches.
            img3 = cv2.drawMatches(
                img1,
                kp1,
                img2,
                kp2,
                matches[:50],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                matchColor=(0, 0, 255),  # Green color for matches
                singlePointColor=None,
                matchesThickness=10,  # Increase the thickness of the match lines
            )

            plt.imshow(img3), plt.show()

        return matches

    elif features == "orb":
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        if visualize:
            # Draw first 10 matches.
            img3 = cv2.drawMatches(
                img1,
                kp1,
                img2,
                kp2,
                matches[:50],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                matchColor=(0, 0, 255),  # Green color for matches
                singlePointColor=None,
                matchesThickness=10,  # Increase the thickness of the match lines
            )

            plt.imshow(img3), plt.show()

        return matches
