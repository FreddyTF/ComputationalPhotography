import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Literal, get_args
from border import apply_border, BorderType
from filter import (
    gaussian_1d,
    gaussian_2d,
    apply_non_separable_filter,
    apply_separable_filter,
)


def compare_kernel_size_and_sigma():
    """
    Compare the effect of different kernel sizes and sigma values on an image.
    """
    # choose a sample image to perform the filtering
    image = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\image_quadratic\Image.jpg"
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kernel_sizes = [1, 5, 9, 15]
    sigmas = [0.1, 3, 9, 15]

    image_results = [[None] * len(kernel_sizes) for _ in range(len(sigmas))]

    for kernel_size in kernel_sizes:
        for sigma in sigmas:
            # calculate kernel
            kernel = gaussian_2d(sigma, kernel_size)
            # apply filter
            image_results[kernel_sizes.index(kernel_size)][sigmas.index(sigma)] = (
                apply_non_separable_filter(image, kernel)
            )

    # create subplots of size 5 x 5
    fig, axs = plt.subplots(len(kernel_sizes), len(sigmas), figsize=(15, 15))

    for i in range(len(kernel_sizes)):
        for j in range(len(sigmas)):
            axs[i, j].imshow(image_results[i][j])
            axs[i, j].set_title(f"Kernel Size: {kernel_sizes[i]}, Sigma: {sigmas[j]}")
            axs[i, j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_kernel_size_and_sigma():
    """
    Plot the 1D Gaussian kernels for different kernel sizes and sigma values.
    """
    kernel_sizes = [1, 5, 9, 15]
    sigmas = [0.1, 1, 2, 3]

    kernels = [[None] * len(kernel_sizes) for _ in range(len(sigmas))]
    for kernel_size in kernel_sizes:
        for sigma in sigmas:
            kernels[kernel_sizes.index(kernel_size)][sigmas.index(sigma)] = gaussian_1d(
                sigma, kernel_size
            )

    fig, axs = plt.subplots(len(kernel_sizes), len(sigmas), figsize=(12, 12))
    max_value = max(max(kernel) for row in kernels for kernel in row)
    for i in range(len(kernel_sizes)):
        for j in range(len(sigmas)):
            axs[i, j].bar(range(len(kernels[i][j])), kernels[i][j])
            axs[i, j].set_ylim(0, max_value)
            axs[i, j].set_title(f"Kernel Size: {kernel_sizes[i]}, Sigma: {sigmas[j]}")
            axs[i, j].set_xlabel("Index")
            axs[i, j].set_ylabel("Value")

    plt.tight_layout()
    plt.show()
