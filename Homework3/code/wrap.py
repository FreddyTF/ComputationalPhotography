import cv2
import numpy as np
import typing
from homography import apply_homography, invert_homography

def warp(homography, image1, image2):
    # use inverse transform

    # we can compute the inverse homography
    # and use it to avoid the hole fillig problem
    inverted_homography = invert_homography(homography)

    # create a new image with the size of the first image + second image
    # and fill it with undefined values

    # fill the first image into the new image



    # fill the second image into the new image
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            if image2[i, j] == [np.nan, np.nan, np.nan]:
                # get the pixel from inverse warping
                # check if the point is inside the image
                x, y = apply_homography(inverted_homography, (j, i))
                x = int(x)
                y = int(y)
                # check if the point is inside the image
                if x >= 0 and x < image1.shape[1] and y >= 0 and y < image1.shape[0]:
                    # set the pixel to the value of the other image
                    image1[y, x] = image2[i, j]

    return image1