import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_histogramm(hist_data: np.ndarray) -> None:
    # Create bin edges (assuming equal-width bins)
    bin_edges = np.arange(len(hist_data) + 1)

    # Plot histogram
    plt.bar(bin_edges[:-1], hist_data, width=1, edgecolor="black")

    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.show()


def color_histogram(image: np.ndarray) -> np.ndarray:
    """
    Compute the color histogram of an image.
    """
    if len(image.shape) != 3:
        raise ValueError("The image must be color")

    r, g, b = cv2.split(image)

    r_equalized, g_equalized, b_equalized = (
        grayscale_histogram(r),
        grayscale_histogram(g),
        grayscale_histogram(b),
    )

    equalized_image = cv2.merge((r_equalized, g_equalized, b_equalized))

    return equalized_image


def get_histogram(image: np.ndarray) -> np.ndarray:
    """
    Compute the histogram of an image.
    """
    height, width = image.shape
    histogram = np.zeros(256, dtype=np.int32)
    for i in range(height):
        for j in range(width):
            histogram[image[i, j].astype(np.uint8)] += 1

    return histogram


def grayscale_histogram(image: np.ndarray) -> np.ndarray:
    """
    Compute the grayscale histogram of an image.
    """
    if len(image.shape) == 3:
        raise ValueError("The image must be grayscale")

    height, width = image.shape

    image = image.astype(np.uint8)
    equalized_image = np.zeros((height, width, 1), dtype=np.uint8)

    # count the orrucences of each pixel value

    histogram = get_histogram(image)

    cum_histogram = np.zeros(256, dtype=np.int32)
    cum_histogram[0] = histogram[0]
    for i in range(1, len(histogram)):
        cum_histogram[i] = cum_histogram[i - 1] + histogram[i]

    cdf = cum_histogram
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)

    equalized_image = cdf_normalized[image]

    return equalized_image


def get_brightness(image: np.ndarray) -> float:
    """
    Compute the brightness of an image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape

    brightness = np.sum(image) / (height * width)

    return brightness


def get_contrast(image: np.ndarray) -> float:
    """
    Compute the contrast of an image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape

    mean = np.mean(image)
    contrast = np.sum((image - mean) ** 2) / (height * width)

    return contrast





