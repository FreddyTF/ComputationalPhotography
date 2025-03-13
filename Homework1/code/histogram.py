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

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Original images
    axs[0, 0].imshow(cv2.cvtColor(image_color1, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image 1")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(cv2.cvtColor(image_color2, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("Original Image 2")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(cv2.cvtColor(image_color3, cv2.COLOR_BGR2RGB))
    axs[0, 2].set_title("Original Image 3")
    axs[0, 2].axis("off")

    # Equalized images
    axs[1, 0].imshow(cv2.cvtColor(image_color1_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Equalized Image 1")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(cv2.cvtColor(image_color2_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title("Equalized Image 2")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(cv2.cvtColor(image_color3_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 2].set_title("Equalized Image 3")
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


def multiplot_histogramm_gray_images_lecture():
    image_gray1 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\lecture_images\gray1.jpg"
    )
    image_gray2 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\lecture_images\gray2.jpg"
    )
    image_gray3 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\lecture_images\gray3.jpg"
    )

    image_gray1_equalized = color_histogram(image_gray1)
    image_gray2_equalized = color_histogram(image_gray2)
    image_gray3_equalized = color_histogram(image_gray3)

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Original images
    axs[0, 0].imshow(cv2.cvtColor(image_gray1, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image 1")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(cv2.cvtColor(image_gray2, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("Original Image 2")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(cv2.cvtColor(image_gray3, cv2.COLOR_BGR2RGB))
    axs[0, 2].set_title("Original Image 3")
    axs[0, 2].axis("off")

    # Equalized images
    axs[1, 0].imshow(cv2.cvtColor(image_gray1_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Equalized Image 1")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(cv2.cvtColor(image_gray2_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title("Equalized Image 2")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(cv2.cvtColor(image_gray3_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 2].set_title("Equalized Image 3")
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


def multiplot_histogramm_color_images_own():
    image_color1 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images\Image1.jpeg"
    )
    image_color2 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images\Image2.jpeg"
    )
    image_color3 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images\Image3.jpeg"
    )

    image_color1_equalized = color_histogram(image_color1)
    image_color2_equalized = color_histogram(image_color2)
    image_color3_equalized = color_histogram(image_color3)

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Original images
    axs[0, 0].imshow(cv2.cvtColor(image_color1, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image 1")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(cv2.cvtColor(image_color2, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("Original Image 2")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(cv2.cvtColor(image_color3, cv2.COLOR_BGR2RGB))
    axs[0, 2].set_title("Original Image 3")
    axs[0, 2].axis("off")

    # Equalized images
    axs[1, 0].imshow(cv2.cvtColor(image_color1_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Equalized Image 1")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(cv2.cvtColor(image_color2_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title("Equalized Image 2")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(cv2.cvtColor(image_color3_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 2].set_title("Equalized Image 3")
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


def multiplot_histogramm_gray_images_own():
    image_gray1 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images\Image1Gray.jpeg"
    )
    image_gray2 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images\Image2Gray.jpeg"
    )
    image_gray3 = cv2.imread(
        "C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images\Image3Gray.jpeg"
    )

    image_gray1_equalized = color_histogram(image_gray1)
    image_gray2_equalized = color_histogram(image_gray2)
    image_gray3_equalized = color_histogram(image_gray3)

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Original images
    axs[0, 0].imshow(cv2.cvtColor(image_gray1, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image 1")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(cv2.cvtColor(image_gray2, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("Original Image 2")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(cv2.cvtColor(image_gray3, cv2.COLOR_BGR2RGB))
    axs[0, 2].set_title("Original Image 3")
    axs[0, 2].axis("off")

    # Equalized images
    axs[1, 0].imshow(cv2.cvtColor(image_gray1_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Equalized Image 1")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(cv2.cvtColor(image_gray2_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title("Equalized Image 2")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(cv2.cvtColor(image_gray3_equalized, cv2.COLOR_BGR2RGB))
    axs[1, 2].set_title("Equalized Image 3")
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


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


#multiplot_histogramm_gray_images_lecture()
#multiplot_histogramm_color_images_lecture()
#multiplot_histogramm_gray_images_own()
multiplot_histogramm_color_images_own()

