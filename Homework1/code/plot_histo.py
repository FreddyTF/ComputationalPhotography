import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Literal, get_args
from border import apply_border, BorderType
from filter import gaussian_1d, gaussian_2d, gaussian_2d_direct
from histogram import (
    get_histogram,
    grayscale_histogram,
    color_histogram,
    get_brightness,
    get_contrast,
)


def multiplot_histogramm_gray(image: np.ndarray):
    if len(image.shape) == 3:
        raise ValueError("The image must be grayscale")

    histogram_before = get_histogram(image)

    equalized_image = grayscale_histogram(image)

    histogram_after = get_histogram(equalized_image)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Original image
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), cmap="gray")
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    # Histogram of original image
    axs[1, 0].bar(
        np.arange(len(histogram_before)), histogram_before, width=1, edgecolor="black"
    )
    axs[1, 0].set_title("Histogram of Original Image")
    axs[1, 0].set_xlabel("Bins")
    axs[1, 0].set_ylabel("Frequency")

    # Equalized image
    axs[0, 1].imshow(cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB), cmap="gray")
    axs[0, 1].set_title("Equalized Image")
    axs[0, 1].axis("off")

    # Histogram of equalized image
    axs[1, 1].bar(
        np.arange(len(histogram_after)), histogram_after, width=1, edgecolor="black"
    )
    axs[1, 1].set_title("Histogram of Equalized Image")
    axs[1, 1].set_xlabel("Bins")
    axs[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def multiplot_histogramm_color(image: np.ndarray):
    if len(image.shape) != 3:
        raise ValueError("The image must be coloredscale")

    r, g, b = cv2.split(image)

    histogram_before = [get_histogram(image) for image in [r, g, b]]

    equalized_image = color_histogram(image)

    r_equal, g_equal, b_equal = cv2.split(equalized_image)
    histogram_after = [get_histogram(image) for image in [r_equal, g_equal, b_equal]]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Original image
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap="gray")
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    # Histogram of original image

    colors = ("r", "g", "b")
    for i, color in enumerate(colors):
        axs[1, 0].plot(histogram_before[i], color=color)
    axs[1, 0].set_title("Histogram of Original Image")
    axs[1, 0].set_xlabel("Bins")
    axs[1, 0].set_ylabel("Frequency")

    # Equalized image
    axs[0, 1].imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB), cmap="gray")
    axs[0, 1].set_title("Equalized Image")
    axs[0, 1].axis("off")

    # Histogram of equalized image
    colors = ("r", "g", "b")
    for i, color in enumerate(colors):
        axs[1, 1].plot(histogram_after[i], color=color)
    axs[1, 1].set_title("Histogram of Equalized Image")
    axs[1, 1].set_xlabel("Bins")
    axs[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def multiplot_histogramm_color_images_lecture():
    image_color1 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\lecture_images\color1.jpg"
    )
    image_color2 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\lecture_images\color2.jpg"
    )
    image_color3 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\lecture_images\color3.jpg"
    )

    image_color1_equalized = color_histogram(image_color1)
    image_color2_equalized = color_histogram(image_color2)
    image_color3_equalized = color_histogram(image_color3)

    r1, g1, b1 = cv2.split(image_color1)
    r2, g2, b2 = cv2.split(image_color2)
    r3, g3, b3 = cv2.split(image_color3)

    r1_equal, g1_equal, b1_equal = cv2.split(image_color1_equalized)
    r2_equal, g2_equal, b2_equal = cv2.split(image_color2_equalized)
    r3_equal, g3_equal, b3_equal = cv2.split(image_color3_equalized)

    histogram_before1 = [get_histogram(channel) for channel in [r1, g1, b1]]
    histogram_before2 = [get_histogram(channel) for channel in [r2, g2, b2]]
    histogram_before3 = [get_histogram(channel) for channel in [r3, g3, b3]]

    histogram_after1 = [
        get_histogram(channel) for channel in [r1_equal, g1_equal, b1_equal]
    ]
    histogram_after2 = [
        get_histogram(channel) for channel in [r2_equal, g2_equal, b2_equal]
    ]
    histogram_after3 = [
        get_histogram(channel) for channel in [r3_equal, g3_equal, b3_equal]
    ]

    fig, axs = plt.subplots(3, 4, figsize=(18, 12))

    # Original images
    brightness1 = get_brightness(image_color1)
    contrast1 = get_contrast(image_color1)
    axs[0, 0].imshow(cv2.cvtColor(image_color1, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title(
        f"Original Image 1\nBrightness: {brightness1:.2f}, Contrast: {contrast1:.2f}"
    )
    axs[0, 0].axis("off")

    brightness2 = get_brightness(image_color2)
    contrast2 = get_contrast(image_color2)
    axs[1, 0].imshow(cv2.cvtColor(image_color2, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title(
        f"Original Image 2\nBrightness: {brightness2:.2f}, Contrast: {contrast2:.2f}"
    )
    axs[1, 0].axis("off")

    brightness3 = get_brightness(image_color3)
    contrast3 = get_contrast(image_color3)
    axs[2, 0].imshow(cv2.cvtColor(image_color3, cv2.COLOR_BGR2RGB))
    axs[2, 0].set_title(
        f"Original Image 3\nBrightness: {brightness3:.2f}, Contrast: {contrast3:.2f}"
    )
    axs[2, 0].axis("off")

    # Histograms of original images
    colors = ("r", "g", "b")
    for i, color in enumerate(colors):
        axs[0, 1].plot(histogram_before1[i], color=color)
    axs[0, 1].set_title("Histogram of Original Image 1")
    axs[0, 1].set_xlabel("Bins")
    axs[0, 1].set_ylabel("Frequency")

    for i, color in enumerate(colors):
        axs[1, 1].plot(histogram_before2[i], color=color)
    axs[1, 1].set_title("Histogram of Original Image 2")
    axs[1, 1].set_xlabel("Bins")
    axs[1, 1].set_ylabel("Frequency")

    for i, color in enumerate(colors):
        axs[2, 1].plot(histogram_before3[i], color=color)
    axs[2, 1].set_title("Histogram of Original Image 3")
    axs[2, 1].set_xlabel("Bins")
    axs[2, 1].set_ylabel("Frequency")

    # Equalized images
    brightness1_equalized = get_brightness(image_color1_equalized)
    contrast1_equalized = get_contrast(image_color1_equalized)
    axs[0, 2].imshow(cv2.cvtColor(image_color1_equalized, cv2.COLOR_BGR2RGB))
    axs[0, 2].set_title(
        f"Equalized Image 1\nBrightness: {brightness1_equalized:.2f}, Contrast: {contrast1_equalized:.2f}"
    )
    axs[0, 2].axis("off")

    brightness2_equalized = get_brightness(image_color2_equalized)
    contrast2_equalized = get_contrast(image_color2_equalized)
    axs[1, 2].imshow(cv2.cvtColor(image_color2_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 2].set_title(
        f"Equalized Image 2\nBrightness: {brightness2_equalized:.2f}, Contrast: {contrast2_equalized:.2f}"
    )
    axs[1, 2].axis("off")

    brightness3_equalized = get_brightness(image_color3_equalized)
    contrast3_equalized = get_contrast(image_color3_equalized)
    axs[2, 2].imshow(cv2.cvtColor(image_color3_equalized, cv2.COLOR_BGR2RGB))
    axs[2, 2].set_title(
        f"Equalized Image 3\nBrightness: {brightness3_equalized:.2f}, Contrast: {contrast3_equalized:.2f}"
    )
    axs[2, 2].axis("off")

    # Histograms of equalized images
    for i, color in enumerate(colors):
        axs[0, 3].plot(histogram_after1[i], color=color)
    axs[0, 3].set_title("Histogram of Equalized Image 1")
    axs[0, 3].set_xlabel("Bins")
    axs[0, 3].set_ylabel("Frequency")

    for i, color in enumerate(colors):
        axs[1, 3].plot(histogram_after2[i], color=color)
    axs[1, 3].set_title("Histogram of Equalized Image 2")
    axs[1, 3].set_xlabel("Bins")
    axs[1, 3].set_ylabel("Frequency")

    for i, color in enumerate(colors):
        axs[2, 3].plot(histogram_after3[i], color=color)
    axs[2, 3].set_title("Histogram of Equalized Image 3")
    axs[2, 3].set_xlabel("Bins")
    axs[2, 3].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("results/histogram/histogram_color_lecture.png")
    # plt.show()


def multiplot_histogramm_gray_images_lecture():
    image_gray1 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\lecture_images\gray1.jpg",
        cv2.IMREAD_GRAYSCALE,
    )
    image_gray2 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\lecture_images\gray2.jpg",
        cv2.IMREAD_GRAYSCALE,
    )
    image_gray3 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\lecture_images\gray3.jpg",
        cv2.IMREAD_GRAYSCALE,
    )

    image_gray1_equalized = grayscale_histogram(image_gray1)
    image_gray2_equalized = grayscale_histogram(image_gray2)
    image_gray3_equalized = grayscale_histogram(image_gray3)

    histogram_before1 = get_histogram(image_gray1)
    histogram_before2 = get_histogram(image_gray2)
    histogram_before3 = get_histogram(image_gray3)

    histogram_after1 = get_histogram(image_gray1_equalized)
    histogram_after2 = get_histogram(image_gray2_equalized)
    histogram_after3 = get_histogram(image_gray3_equalized)

    fig, axs = plt.subplots(3, 4, figsize=(18, 12))

    # Original images
    brightness1 = get_brightness(image_gray1)
    contrast1 = get_contrast(image_gray1)
    axs[0, 0].imshow(cv2.cvtColor(image_gray1, cv2.COLOR_GRAY2RGB))
    axs[0, 0].set_title(
        f"Original Image 1\nBrightness: {brightness1:.2f}, Contrast: {contrast1:.2f}"
    )
    axs[0, 0].axis("off")

    brightness2 = get_brightness(image_gray2)
    contrast2 = get_contrast(image_gray2)
    axs[1, 0].imshow(cv2.cvtColor(image_gray2, cv2.COLOR_GRAY2RGB))
    axs[1, 0].set_title(
        f"Original Image 2 \nBrightness: {brightness2:.2f}, Contrast: {contrast2:.2f}"
    )
    axs[1, 0].axis("off")

    brightness3 = get_brightness(image_gray3)
    contrast3 = get_contrast(image_gray3)
    axs[2, 0].imshow(cv2.cvtColor(image_gray3, cv2.COLOR_GRAY2RGB))
    axs[2, 0].set_title(
        f"Original Image 3 \nBrightness: {brightness3:.2f}, Contrast: {contrast3:.2f}"
    )
    axs[2, 0].axis("off")

    # Histograms of original images
    axs[0, 1].bar(
        np.arange(len(histogram_before1)), histogram_before1, width=1, edgecolor="black"
    )
    axs[0, 1].set_title("Histogram of Original Image 1")
    axs[0, 1].set_xlabel("Bins")
    axs[0, 1].set_ylabel("Frequency")

    axs[1, 1].bar(
        np.arange(len(histogram_before2)), histogram_before2, width=1, edgecolor="black"
    )
    axs[1, 1].set_title("Histogram of Original Image 2")
    axs[1, 1].set_xlabel("Bins")
    axs[1, 1].set_ylabel("Frequency")

    axs[2, 1].bar(
        np.arange(len(histogram_before3)), histogram_before3, width=1, edgecolor="black"
    )
    axs[2, 1].set_title("Histogram of Original Image 3")
    axs[2, 1].set_xlabel("Bins")
    axs[2, 1].set_ylabel("Frequency")

    # Equalized images
    brightness1_equalized = get_brightness(image_gray1_equalized)
    contrast1_equalized = get_contrast(image_gray1_equalized)
    axs[0, 2].imshow(cv2.cvtColor(image_gray1_equalized, cv2.COLOR_GRAY2RGB))
    axs[0, 2].set_title(
        f"Equalized Image 1\nBrightness: {brightness1_equalized:.2f}, Contrast: {contrast1_equalized:.2f}"
    )
    axs[0, 2].axis("off")

    brightness2_equalized = get_brightness(image_gray2_equalized)
    contrast2_equalized = get_contrast(image_gray2_equalized)
    axs[1, 2].imshow(cv2.cvtColor(image_gray2_equalized, cv2.COLOR_GRAY2RGB))
    axs[1, 2].set_title(
        f"Equalized Image 2\nBrightness: {brightness2_equalized:.2f}, Contrast: {contrast2_equalized:.2f}"
    )
    axs[1, 2].axis("off")

    brightness3_equalized = get_brightness(image_gray3_equalized)
    contrast3_equalized = get_contrast(image_gray3_equalized)
    axs[2, 2].imshow(cv2.cvtColor(image_gray3_equalized, cv2.COLOR_GRAY2RGB))
    axs[2, 2].set_title(
        f"Equalized Image 3\nBrightness: {brightness3_equalized:.2f}, Contrast: {contrast3_equalized:.2f}"
    )
    axs[2, 2].axis("off")

    # Histograms of equalized images
    axs[0, 3].bar(
        np.arange(len(histogram_after1)), histogram_after1, width=1, edgecolor="black"
    )
    axs[0, 3].set_title("Histogram of Equalized Image 1")
    axs[0, 3].set_xlabel("Bins")
    axs[0, 3].set_ylabel("Frequency")

    axs[1, 3].bar(
        np.arange(len(histogram_after2)), histogram_after2, width=1, edgecolor="black"
    )
    axs[1, 3].set_title("Histogram of Equalized Image 2")
    axs[1, 3].set_xlabel("Bins")
    axs[1, 3].set_ylabel("Frequency")

    axs[2, 3].bar(
        np.arange(len(histogram_after3)), histogram_after3, width=1, edgecolor="black"
    )
    axs[2, 3].set_title("Histogram of Equalized Image 3")
    axs[2, 3].set_xlabel("Bins")
    axs[2, 3].set_ylabel("Frequency")

    plt.tight_layout()
    # plt.show()
    plt.savefig("results/histogram/histogram_gray_lecture.png")


def multiplot_histogramm_color_images_own():
    image_color1 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images_small\Image1.jpg"
    )
    image_color2 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images_small\Image2.jpg"
    )
    image_color3 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images_small\Image3.jpg"
    )

    image_color1_equalized = color_histogram(image_color1)
    image_color2_equalized = color_histogram(image_color2)
    image_color3_equalized = color_histogram(image_color3)

    r1, g1, b1 = cv2.split(image_color1)
    r2, g2, b2 = cv2.split(image_color2)
    r3, g3, b3 = cv2.split(image_color3)

    r1_equal, g1_equal, b1_equal = cv2.split(image_color1_equalized)
    r2_equal, g2_equal, b2_equal = cv2.split(image_color2_equalized)
    r3_equal, g3_equal, b3_equal = cv2.split(image_color3_equalized)

    histogram_before1 = [get_histogram(channel) for channel in [r1, g1, b1]]
    histogram_before2 = [get_histogram(channel) for channel in [r2, g2, b2]]
    histogram_before3 = [get_histogram(channel) for channel in [r3, g3, b3]]

    histogram_after1 = [
        get_histogram(channel) for channel in [r1_equal, g1_equal, b1_equal]
    ]
    histogram_after2 = [
        get_histogram(channel) for channel in [r2_equal, g2_equal, b2_equal]
    ]
    histogram_after3 = [
        get_histogram(channel) for channel in [r3_equal, g3_equal, b3_equal]
    ]

    fig, axs = plt.subplots(3, 4, figsize=(18, 12))

    # Original images
    brightness1 = get_brightness(image_color1)
    contrast1 = get_contrast(image_color1)
    axs[0, 0].imshow(cv2.cvtColor(image_color1, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title(
        f"Original Image 1\nBrightness: {brightness1:.2f}, Contrast: {contrast1:.2f}"
    )
    axs[0, 0].axis("off")

    brightness2 = get_brightness(image_color2)
    contrast2 = get_contrast(image_color2)
    axs[1, 0].imshow(cv2.cvtColor(image_color2, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title(
        f"Original Image 2\nBrightness: {brightness2:.2f}, Contrast: {contrast2:.2f}"
    )
    axs[1, 0].axis("off")

    brightness3 = get_brightness(image_color3)
    contrast3 = get_contrast(image_color3)
    axs[2, 0].imshow(cv2.cvtColor(image_color3, cv2.COLOR_BGR2RGB))
    axs[2, 0].set_title(
        f"Original Image 3\nBrightness: {brightness3:.2f}, Contrast: {contrast3:.2f}"
    )
    axs[2, 0].axis("off")

    # Histograms of original images
    colors = ("r", "g", "b")
    for i, color in enumerate(colors):
        axs[0, 1].plot(histogram_before1[i], color=color)
    axs[0, 1].set_title("Histogram of Original Image 1")
    axs[0, 1].set_xlabel("Bins")
    axs[0, 1].set_ylabel("Frequency")

    for i, color in enumerate(colors):
        axs[1, 1].plot(histogram_before2[i], color=color)
    axs[1, 1].set_title("Histogram of Original Image 2")
    axs[1, 1].set_xlabel("Bins")
    axs[1, 1].set_ylabel("Frequency")

    for i, color in enumerate(colors):
        axs[2, 1].plot(histogram_before3[i], color=color)
    axs[2, 1].set_title("Histogram of Original Image 3")
    axs[2, 1].set_xlabel("Bins")
    axs[2, 1].set_ylabel("Frequency")

    # Equalized images
    brightness1_equalized = get_brightness(image_color1_equalized)
    contrast1_equalized = get_contrast(image_color1_equalized)
    axs[0, 2].imshow(cv2.cvtColor(image_color1_equalized, cv2.COLOR_BGR2RGB))
    axs[0, 2].set_title(
        f"Equalized Image 1\nBrightness: {brightness1_equalized:.2f}, Contrast: {contrast1_equalized:.2f}"
    )
    axs[0, 2].axis("off")

    brightness2_equalized = get_brightness(image_color2_equalized)
    contrast2_equalized = get_contrast(image_color2_equalized)
    axs[1, 2].imshow(cv2.cvtColor(image_color2_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 2].set_title(
        f"Equalized Image 2\nBrightness: {brightness2_equalized:.2f}, Contrast: {contrast2_equalized:.2f}"
    )
    axs[1, 2].axis("off")

    brightness3_equalized = get_brightness(image_color3_equalized)
    contrast3_equalized = get_contrast(image_color3_equalized)
    axs[2, 2].imshow(cv2.cvtColor(image_color3_equalized, cv2.COLOR_BGR2RGB))
    axs[2, 2].set_title(
        f"Equalized Image 3\nBrightness: {brightness3_equalized:.2f}, Contrast: {contrast3_equalized:.2f}"
    )
    axs[2, 2].axis("off")

    # Histograms of equalized images
    for i, color in enumerate(colors):
        axs[0, 3].plot(histogram_after1[i], color=color)
    axs[0, 3].set_title("Histogram of Equalized Image 1")
    axs[0, 3].set_xlabel("Bins")
    axs[0, 3].set_ylabel("Frequency")

    for i, color in enumerate(colors):
        axs[1, 3].plot(histogram_after2[i], color=color)
    axs[1, 3].set_title("Histogram of Equalized Image 2")
    axs[1, 3].set_xlabel("Bins")
    axs[1, 3].set_ylabel("Frequency")

    for i, color in enumerate(colors):
        axs[2, 3].plot(histogram_after3[i], color=color)
    axs[2, 3].set_title("Histogram of Equalized Image 3")
    axs[2, 3].set_xlabel("Bins")
    axs[2, 3].set_ylabel("Frequency")

    plt.tight_layout()
    # plt.show()
    plt.savefig("results/histogram/histogram_color_own.png")


def multiplot_histogramm_gray_images_own():
    image_gray1 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images_small\Image1Gray.jpg",
        cv2.IMREAD_GRAYSCALE,
    )
    image_gray2 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images_small\Image2Gray.jpg",
        cv2.IMREAD_GRAYSCALE,
    )
    image_gray3 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images_small\Image3Gray.jpg",
        cv2.IMREAD_GRAYSCALE,
    )

    image_gray1_equalized = grayscale_histogram(image_gray1)
    image_gray2_equalized = grayscale_histogram(image_gray2)
    image_gray3_equalized = grayscale_histogram(image_gray3)

    histogram_before1 = get_histogram(image_gray1)
    histogram_before2 = get_histogram(image_gray2)
    histogram_before3 = get_histogram(image_gray3)

    histogram_after1 = get_histogram(image_gray1_equalized)
    histogram_after2 = get_histogram(image_gray2_equalized)
    histogram_after3 = get_histogram(image_gray3_equalized)

    fig, axs = plt.subplots(3, 4, figsize=(18, 12))

    # Original images
    brightness1 = get_brightness(image_gray1)
    contrast1 = get_contrast(image_gray1)
    axs[0, 0].imshow(cv2.cvtColor(image_gray1, cv2.COLOR_GRAY2RGB))
    axs[0, 0].set_title(
        f"Original Image 1\nBrightness: {brightness1:.2f}, Contrast: {contrast1:.2f}"
    )
    axs[0, 0].axis("off")

    brightness2 = get_brightness(image_gray2)
    contrast2 = get_contrast(image_gray2)
    axs[1, 0].imshow(cv2.cvtColor(image_gray2, cv2.COLOR_GRAY2RGB))
    axs[1, 0].set_title(
        f"Original Image 2 \nBrightness: {brightness2:.2f}, Contrast: {contrast2:.2f}"
    )
    axs[1, 0].axis("off")

    brightness3 = get_brightness(image_gray3)
    contrast3 = get_contrast(image_gray3)
    axs[2, 0].imshow(cv2.cvtColor(image_gray3, cv2.COLOR_GRAY2RGB))
    axs[2, 0].set_title(
        f"Original Image 3 \nBrightness: {brightness3:.2f}, Contrast: {contrast3:.2f}"
    )
    axs[2, 0].axis("off")

    # Histograms of original images
    axs[0, 1].bar(
        np.arange(len(histogram_before1)), histogram_before1, width=1, edgecolor="black"
    )
    axs[0, 1].set_title("Histogram of Original Image 1")
    axs[0, 1].set_xlabel("Bins")
    axs[0, 1].set_ylabel("Frequency")

    axs[1, 1].bar(
        np.arange(len(histogram_before2)), histogram_before2, width=1, edgecolor="black"
    )
    axs[1, 1].set_title("Histogram of Original Image 2")
    axs[1, 1].set_xlabel("Bins")
    axs[1, 1].set_ylabel("Frequency")

    axs[2, 1].bar(
        np.arange(len(histogram_before3)), histogram_before3, width=1, edgecolor="black"
    )
    axs[2, 1].set_title("Histogram of Original Image 3")
    axs[2, 1].set_xlabel("Bins")
    axs[2, 1].set_ylabel("Frequency")

    # Equalized images
    brightness1_equalized = get_brightness(image_gray1_equalized)
    contrast1_equalized = get_contrast(image_gray1_equalized)
    axs[0, 2].imshow(cv2.cvtColor(image_gray1_equalized, cv2.COLOR_GRAY2RGB))
    axs[0, 2].set_title(
        f"Equalized Image 1\nBrightness: {brightness1_equalized:.2f}, Contrast: {contrast1_equalized:.2f}"
    )
    axs[0, 2].axis("off")

    brightness2_equalized = get_brightness(image_gray2_equalized)
    contrast2_equalized = get_contrast(image_gray2_equalized)
    axs[1, 2].imshow(cv2.cvtColor(image_gray2_equalized, cv2.COLOR_GRAY2RGB))
    axs[1, 2].set_title(
        f"Equalized Image 2\nBrightness: {brightness2_equalized:.2f}, Contrast: {contrast2_equalized:.2f}"
    )
    axs[1, 2].axis("off")

    brightness3_equalized = get_brightness(image_gray3_equalized)
    contrast3_equalized = get_contrast(image_gray3_equalized)
    axs[2, 2].imshow(cv2.cvtColor(image_gray3_equalized, cv2.COLOR_GRAY2RGB))
    axs[2, 2].set_title(
        f"Equalized Image 3\nBrightness: {brightness3_equalized:.2f}, Contrast: {contrast3_equalized:.2f}"
    )
    axs[2, 2].axis("off")

    # Histograms of equalized images
    axs[0, 3].bar(
        np.arange(len(histogram_after1)), histogram_after1, width=1, edgecolor="black"
    )
    axs[0, 3].set_title("Histogram of Equalized Image 1")
    axs[0, 3].set_xlabel("Bins")
    axs[0, 3].set_ylabel("Frequency")

    axs[1, 3].bar(
        np.arange(len(histogram_after2)), histogram_after2, width=1, edgecolor="black"
    )
    axs[1, 3].set_title("Histogram of Equalized Image 2")
    axs[1, 3].set_xlabel("Bins")
    axs[1, 3].set_ylabel("Frequency")

    axs[2, 3].bar(
        np.arange(len(histogram_after3)), histogram_after3, width=1, edgecolor="black"
    )
    axs[2, 3].set_title("Histogram of Equalized Image 3")
    axs[2, 3].set_xlabel("Bins")
    axs[2, 3].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("results/histogram/histogram_gray_own.png")
    # plt.show()


def create_plots_for_report():
    image_color = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\lecture_images\color1.jpg"
    )
    multiplot_histogramm_color(image_color)

    image_gray = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\lecture_images\gray1.jpg",
        cv2.IMREAD_GRAYSCALE,
    )
    multiplot_histogramm_gray(image_gray)
