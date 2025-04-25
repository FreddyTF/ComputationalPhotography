import cv2
import numpy as np
import typing
from homography import apply_homography, invert_homography
from matplotlib import pyplot as plt
import time


def create_mask(
    image1, version, smoothing_window_size, height_panorama, width_panorama
):
    """Creates the mask using query and train images for blending the images,
    using a gaussian smoothing window/kernel

    Args:
        image1 (numpy array)
        image2 (numpy array)
        version (str) == 'left_image' or 'right_image'

    Returns:
        masks
    """

    offset = int(smoothing_window_size / 2)
    barrier = image1.shape[1] - int(smoothing_window_size / 2)
    mask = np.zeros((height_panorama, width_panorama))

    if version == "left_image":
        mask[:, barrier - offset : barrier + offset] = np.tile(
            np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)
        )
        mask[:, : barrier - offset] = 1
    else:
        mask[:, barrier - offset : barrier + offset] = np.tile(
            np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)
        )
        mask[:, barrier + offset :] = 1
    result = cv2.merge([mask, mask, mask])
    plt.imshow(result)
    plt.savefig(
        r"report\images\8\_" + time.strftime("%Y-%m-%d_%H-%M-%S") + ".png",
        dpi=300,  # high resolution
    )
    plt.close()

    return result


def warp_and_stich_images(
    homography: np.ndarray, image1: np.ndarray, image2: np.ndarray
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

    if homography[0, 2] > 0:
        image1, image2 = image2, image1
        homography = invert_homography(homography)

    homography = invert_homography(homography)

    height_img1 = image1.shape[0]
    width_img1 = image1.shape[1]
    width_img2 = image2.shape[1]

    lowest_width = min(width_img1, width_img2)
    smoothing_window_percent = 0.10
    smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000))

    height_panorama = height_img1
    width_panorama = width_img1 + width_img2

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = create_mask(
        image1,
        version="left_image",
        smoothing_window_size=smoothing_window_size,
        height_panorama=height_panorama,
        width_panorama=width_panorama,
    )

    mask2 = create_mask(
        image1,
        version="right_image",
        smoothing_window_size=smoothing_window_size,
        height_panorama=height_panorama,
        width_panorama=width_panorama,
    )

    panorama1[0 : image1.shape[0], 0 : image1.shape[1], :] = image1
    panorama1 *= mask1

    panorama2 = (
        cv2.warpPerspective(image2, homography, (width_panorama, height_panorama))
        * mask2
    )
    result = panorama1 + panorama2

    # remove extra blackspace
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1

    final_result = result[min_row:max_row, min_col:max_col, :]
    final_result = np.clip(final_result, 0, 255).astype(np.uint8)
    plt.imshow(final_result)
    plt.savefig(
        r"report\images\8\_" + time.strftime("%Y-%m-%d_%H-%M-%S") + ".png",
        dpi=300,  # high resolution
    )
    plt.close()

    return final_result
