from filter import filterGaussian
from helpers import readImagesToNumpyArray
from pathlib import Path

import cv2


def main():
    # Load the image

    path_to_workspace = Path(
        r"C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography"
    )
    path_subfolder = Path(r"Homework1\images")
    path_to_images = path_to_workspace / path_subfolder
    images = readImagesToNumpyArray(str(path_to_images))
    first_image = images[0]
    processed_image = filterGaussian(first_image, 3, 1, cv2.BORDER_CONSTANT, False)

    cv2.imshow("Original Image", first_image)
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
