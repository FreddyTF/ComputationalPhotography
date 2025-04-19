import cv2
import numpy as np
import typing
from homography import apply_homography, invert_homography
from matplotlib import pyplot as plt


def warp_images_inverse(
    homography: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray,
    visualize: bool = False,
) -> np.ndarray:
    """
    Warp image2 to the perspective of image1 using the provided homography matrix.

    Parameters:
        homography (np.ndarray): The homography matrix to warp image2.
        image1 (np.ndarray): The first image (base image).
        image2 (np.ndarray): The second image to be warped.

    Returns:
        np.ndarray: The warped panorama image.
    """
    # if homography[0, 2] < 0:
    #     image1, image2 = image2, image1
    #     homography = invert_homography(homography)
    if homography[0, 2] > 0:
        image1, image2 = image2, image1
        homography = invert_homography(homography)

    inverted_homography = invert_homography(homography)

    # Warp image2 into the coordinate space of image1
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Get corners of image2 before warp
    corners_img2 = np.float32(
        [[0, 0], [0, height2], [width2, height2], [width2, 0]]
    ).reshape(-1, 1, 2)

    H = homography
    transformed_corners = cv2.perspectiveTransform(corners_img2, H)

    # Combine corners from both images to compute bounding box
    corners_img1 = np.float32(
        [[0, 0], [0, height1], [width1, height1], [width1, 0]]
    ).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners_img1, transformed_corners), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Size of the final panorama
    panorama_size = (x_max - x_min, y_max - y_min)

    # Adjust homography to include translation

    # Warp image2

    # translation = np.array([[1, 0, x_min], [0, 1, y_min], [0, 0, 1]])
    # translation @

    warped_img2 = cv2.warpPerspective(
        image2,
        inverted_homography,
        panorama_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    # Place image1 on the panorama canvas
    panorama = np.zeros((panorama_size[1], panorama_size[0], 3), dtype=np.uint8)
    # panorama[y_min : height1 + y_min, x_min : width1 + x_min] = image1
    # panorama[-y_min : -y_min + height1, -x_min : -x_min + width1] = image1
    panorama[0 : 0 + height1, 0 : 0 + width1] = image1

    if visualize:
        plt.imshow(panorama)
        plt.title("Panorama")
        plt.show()

        plt.imshow(warped_img2)
        plt.title("Warped Image 2")
        plt.show()

    # Blend using mask
    mask = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask_3ch = cv2.merge([mask, mask, mask])
    mask_inv_3ch = cv2.merge([mask_inv, mask_inv, mask_inv])

    panorama_bg = cv2.bitwise_and(panorama, mask_inv_3ch)
    img2_fg = cv2.bitwise_and(warped_img2, mask_3ch)
    final_blend = cv2.add(panorama_bg, img2_fg)

    plt.imshow(final_blend)
    plt.title("Final Blended Panorama")
    plt.show()

    return final_blend
