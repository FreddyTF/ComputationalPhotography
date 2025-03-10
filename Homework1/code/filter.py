import cv2
import typing
from typing import Literal, get_args
import numpy as np


BorderType = Literal[
    cv2.BORDER_CONSTANT,
    cv2.BORDER_REPLICATE,
    cv2.BORDER_REFLECT,
    cv2.BORDER_WRAP,
    cv2.BORDER_REFLECT_101,
]


def gaussian_1d(sigma: int, size: int) -> np.ndarray:
    """
    Create a 1D Gaussian kernel.
    :param sigma: The sigma value to use for the kernel  (standard deviation)
    :param size: The size of the kernel to use
    """

    kernel = np.zeros(size)

    center = size // 2

    for i in range(0, size):
        kernel[i] = (
            1
            / (np.sqrt(2 * np.pi) * sigma)
            * np.exp(-((i - center) ** 2) / (2 * (sigma**2)))
        )

    # normalize

    kernel = kernel / np.sum(kernel)

    return kernel


def filterGaussian(
    image: np.ndarray,
    kernel_size: int,
    kernel_sigma: int,
    border_type: BorderType,
    separable: bool,
) -> np.ndarray:
    """
    Apply a Gaussian filter to an image.
    :param image: The image to filter
    :param kernel_size: The size of the kernel to use
    :param kernel_sigma: The sigma value to use for the kernel  (standard deviation)
    :param border_type: The border type to use
    :param separable: true if separation of filter shall be used, false if normal 2d convolution operation shall be used
    :return: The filtered image
    """

    width, height = image.shape[:2]

    # check inputs according to taks definition
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number")
    if kernel_size > width or kernel_size > height:
        raise ValueError("Kernel size must be smaller than image dimensions")

    if kernel_sigma <= 0:
        raise ValueError("Sigma must be greater than 0")

    if border_type not in get_args(BorderType):
        raise ValueError("Border Type not valid")

    # testing config:
    kernel_size = 3
    separable = True

    if separable:
        # apply separable filter
        # 1d convolutional filter

        pass

    else:
        # 2d convolutional filter
        pass

    return image
