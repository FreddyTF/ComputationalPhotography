from filter import (
    filterGaussian,
)
from plot_filter import (
    compare_kernel_size_and_sigma,
    plot_kernel_size_and_sigma,
)
from border import (
    apply_border,
    compare_border_types,
    BorderType,
)

from plot_histo import (
    multiplot_histogramm_gray_images_lecture,
    multiplot_histogramm_color_images_lecture,
    multiplot_histogramm_gray_images_own,
    multiplot_histogramm_color_images_own,
)
from helpers import readImagesToNumpyArray
from pathlib import Path
from histogram import (
    grayscale_histogram,
    color_histogram,
)


import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    # Load the image

    if True:
        # load lecture_images
        path_to_workspace = Path(
            r"C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography"
        )
        path_subfolder = Path(r"Homework1\images\lecture_images")
        path_to_images = path_to_workspace / path_subfolder

        images = readImagesToNumpyArray(str(path_to_images))

        i = 0
        for image in images:
            processed_image = filterGaussian(
                image=image,
                kernel_size=5,
                kernel_sigma=3,
                border_type=cv2.BORDER_CONSTANT,
                separable=False,
            )

            # create subplot for original and processed image
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # plot original image
            axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            # plot processed image
            axs[1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            axs[1].set_title("Processed Image")
            axs[1].axis("off")

            plt.tight_layout()
            path_subfolder = Path(r"Homework1\results\histogram\output_lecture")
            save_path = path_to_workspace / path_subfolder
            plt.savefig(save_path / f"output{i}.png")
            i += 1

    if True:
        # load lecture_images
        path_to_workspace = Path(
            r"C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography"
        )
        path_subfolder = Path(r"Homework1\images\own_images_small")
        path_to_images = path_to_workspace / path_subfolder

        images = readImagesToNumpyArray(str(path_to_images))

        i = 0
        for image in images:
            processed_image = filterGaussian(
                image=image,
                kernel_size=5,
                kernel_sigma=3,
                border_type=cv2.BORDER_CONSTANT,
                separable=False,
            )

            # create subplot for original and processed image
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # plot original image
            axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            # plot processed image
            axs[1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            axs[1].set_title("Processed Image")
            axs[1].axis("off")

            plt.tight_layout()
            path_subfolder = Path(r"Homework1\results\gaussian\output_own_small")
            save_path = path_to_workspace / path_subfolder
            plt.savefig(save_path / f"output{i}.png")
            i += 1

    if True:
        # load lecture_images
        path_to_workspace = Path(
            r"C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography"
        )
        path_subfolder = Path(r"Homework1\images\lecture_images")
        path_to_images = path_to_workspace / path_subfolder
        images = readImagesToNumpyArray(str(path_to_images))
        first_image = images[0]
        # print(first_image.shape)
        processed_image = filterGaussian(
            image=first_image,
            kernel_size=3,
            kernel_sigma=0.01,
            border_type=cv2.BORDER_CONSTANT,
            separable=True,
        )

        cv2.imshow("Original Image", first_image)
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)

    if True:
        # example image test
        path_to_workspace = Path(
            r"C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography"
        )
        path_subfolder = Path(r"Homework1\images\own_images_small")
        path_to_images = path_to_workspace / path_subfolder
        images = readImagesToNumpyArray(str(path_to_images))
        first_image = np.zeros((1, 1, 1))
        for image in images:
            # Split channels
            b, g, r = cv2.split(image)

            # Check if all channels are equal
            if np.array_equal(b, g) and np.array_equal(g, r):
                # print("Found grayscale image")
                first_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                break

        first_image = images[0]

        processed_image = color_histogram(first_image)
        # processed_image = grayscale_histogram(first_image)
        cv2.imshow("Original Image", first_image)
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
    if True:
        # make kernel comparisons
        compare_kernel_size_and_sigma()
        plot_kernel_size_and_sigma()

    if True:
        # plot histogram comparison
        multiplot_histogramm_gray_images_lecture()
        multiplot_histogramm_color_images_lecture()
        multiplot_histogramm_gray_images_own()
        multiplot_histogramm_color_images_own()


if __name__ == "__main__":
    main()
