import cv2
import numpy as np
import typing
from homography import apply_homography, invert_homography
from matplotlib import pyplot as plt
import time


def warp_and_stich_images(
    homography: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray,
    blending: str = "poisson",
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

    warped_img2 = cv2.warpPerspective(
        image2,
        inverted_homography,
        panorama_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    # Place image1 on the panorama canvas
    panorama = np.zeros((panorama_size[1], panorama_size[0], 3), dtype=np.uint8)
    panorama[0:height1, 0:width1] = image1

    if False:
        plt.imshow(panorama)
        plt.title("Panorama")
        plt.show()

        plt.imshow(warped_img2)
        plt.title("Warped Image 2")
        plt.show()

    if blending == "copy_paste":
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
    elif blending == "alpha_blending":
        mask = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = mask.astype(float) / 255.0

        # Blend images using the mask as weights
        alpha = 0.5
        beta = 1.0 - alpha
        final_blend = cv2.addWeighted(warped_img2, alpha, panorama, beta, 0.0)

        exclude_1 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
        exclude_2 = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        _, exclude_1 = cv2.threshold(exclude_1, 0, 255, cv2.THRESH_BINARY)
        _, exclude_2 = cv2.threshold(exclude_2, 0, 255, cv2.THRESH_BINARY)
        exclude_1_3ch = cv2.merge([exclude_1, exclude_1, exclude_1])
        exclude_2_3ch = cv2.merge([exclude_2, exclude_2, exclude_2])
        exclude_mask = cv2.bitwise_and(exclude_1_3ch, exclude_2_3ch)
        exclude_mask = cv2.bitwise_not(exclude_mask)
        warped_exluced_1 = cv2.bitwise_and(warped_img2, exclude_mask)
        panorame_excluded_2 = cv2.bitwise_and(panorama, exclude_mask)
        final_blend = cv2.addWeighted(warped_exluced_1, alpha, final_blend, 1.0, 0.0)
        final_blend = cv2.addWeighted(panorame_excluded_2, beta, final_blend, 1.0, 0.0)

        final_blend = np.clip(final_blend, 0, 255).astype(np.uint8)
    elif blending == "multi_band_blending":
        pass

    elif blending == "poisson":
        # Create a mask for the source image (warped_img2)
        mask = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

        # Find center point for blending (adjust as needed)
        # This is where the center of warped_img2 will be placed in the panorama
        ys, xs = np.where(mask > 0)
        center = (int(np.mean(xs)), int(np.mean(ys)))
        # center = (panorama.shape[1] // 2, panorama.shape[0] // 2)

        # Poisson blending using seamlessClone
        final_blend = cv2.seamlessClone(
            warped_img2,  # source
            panorama,  # destination
            mask,  # mask
            center,  # center position (tuple)
            cv2.MIXED_CLONE,  # or cv2.MIXED_CLONE for mixed gradients
        )
    elif blending == "copy_paste_2":
        # Get the dimensions of the images
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        # Get the canvas dimesions
        pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        img2_warped = cv2.warpPerspective(image2, H, (w1 + w2, h1))

        # Place the first image on the canvas
        img2_warped[0:h1, 0:w1] = image1

    else:
        raise ValueError(
            "Invalid blending method. Choose 'alpha_blending', 'multi_band_blending', or 'poisson'."
        )

    plt.imshow(final_blend)
    plt.title("Final Blended Panorama")
    plt.show()

    return final_blend


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


def warp_and_stich_images_2(
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
