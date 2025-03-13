from filter import filterGaussian
from helpers import readImagesToNumpyArray
from pathlib import Path
from histogram import grayscale_histogram, color_histogram

import cv2
import numpy as np


def main():
    # Load the image

    if False:
        path_to_workspace = Path(
            r"C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography"
        )
        path_subfolder = Path(r"Homework1\images")
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
        path_to_workspace = Path(
            r"C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography"
        )
        path_subfolder = Path(r"Homework1\images")
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


if __name__ == "__main__":
    main()
