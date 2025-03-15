import cv2
import typing
from typing import Literal, get_args
import numpy as np
from border import apply_border, BorderType
from matplotlib import pyplot as plt


def gaussian_1d(sigma: float, size: int) -> np.ndarray:
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


def gaussian_2d(sigma: float, size: int) -> np.ndarray:
    """
    Create a 2D Gaussian kernel.
    :param sigma: The sigma value to use for the kernel  (standard deviation)
    :param size: The size of the kernel to use
    """
    kernel_1d = gaussian_1d(sigma, size)
    kernel = np.outer(kernel_1d, kernel_1d)

    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_2d_direct(sigma: float, size: int) -> np.ndarray:
    """
    Create a 2D Gaussian kernel.
    :param sigma: The sigma value to use for the kernel  (standard deviation)
    :param size: The size of the kernel to use
    """
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(0, size):
        for j in range(0, size):
            kernel[i, j] = (
                1
                / (2 * np.pi * sigma**2)
                * np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * (sigma**2)))
            )

    # normalize
    kernel = kernel / np.sum(kernel)

    return kernel


def apply_separable_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a separable filter to an image.
    :param image: The image to filter
    :param kernel: The kernel to use
    :return: The filtered image
    """
    assert kernel.ndim == 1, "Kernel must be a 1D array."

    filter_size = kernel.shape[0]
    height, width = image.shape[:2]

    output_width = width - filter_size + 1
    output_height = height - filter_size + 1

    intermediate_image = np.zeros((height, output_width, 3))

    output_image = np.zeros((output_height, output_width, 3))

    # apply horizontal filter
    for k in range(3):
        # horizontal filter
        for i in range(0, height):
            for j in range(0, output_width):
                intermediate_image[i, j, k] = np.sum(
                    image[i, j : j + filter_size, k] * kernel
                )

        # apply vertical filter
        for i in range(0, output_height):
            for j in range(0, output_width):
                output_image[i, j, k] = np.sum(
                    intermediate_image[i : i + filter_size, j, k] * kernel
                )

    # check inputs according to taks definition

    return output_image.astype(np.uint8)


def apply_non_separable_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a separable filter to an image.
    :param image: The image to filter
    :param kernel: The kernel to use
    :return: The filtered image
    """

    # print(f"{kernel = }")

    filter_size = kernel.shape[0]
    height, width = image.shape[:2]

    output_height = height - filter_size + 1
    output_width = width - filter_size + 1

    output_image = np.zeros((output_height, output_width, 3))

    for k in range(3):
        for i in range(0, output_height):
            for j in range(0, output_width):
                output_image[i, j, k] = np.sum(
                    kernel * image[i : i + filter_size, j : j + filter_size, k]
                )

    return output_image.astype(np.uint8)


def filterGaussian(
    image: np.ndarray,
    kernel_size: int,
    kernel_sigma: float,
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

    image = apply_border(image, border_type)

    if separable:
        # apply separable filter
        kernel_1d = gaussian_1d(kernel_sigma, kernel_size)
        # 1d convolutional filter
        # print(f"{kernel_1d = }")
        image = apply_separable_filter(image, kernel_1d)

    else:
        # apply separable filter
        kernel_2d = gaussian_2d(kernel_sigma, kernel_size)
        # 2d convolutional filter
        image = apply_non_separable_filter(image, kernel_2d)

    return image
