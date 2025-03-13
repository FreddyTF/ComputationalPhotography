from typing import Literal, get_args

import cv2
import numpy as np

BorderType = Literal[
    cv2.BORDER_CONSTANT,
    cv2.BORDER_REPLICATE,
    cv2.BORDER_REFLECT,
    cv2.BORDER_WRAP,
    cv2.BORDER_REFLECT_101,
]


def apply_border(image: np.ndarray, border_type: BorderType, border_size: int = 1) -> np.ndarray:
    """
    Apply a border to an image.
    :param image: The image to filter
    :param border_type: The border type to use
    :return: The image with added border
    """
    return cv2.copyMakeBorder(image, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=border_type)


def compare_border_types(image: np.ndarray, border_size: int) -> None:
    """
    Compare different border types.
    :param image: The image to filter
    """
    border_types = get_args(BorderType)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(border_types), figsize=(15, 5))
    for ax, border_type in zip(axes, border_types):
        processed_image = apply_border(image, border_type, border_size=border_size)
        ax.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        border_type_name = {
            cv2.BORDER_CONSTANT: "BORDER_CONSTANT",
            cv2.BORDER_REPLICATE: "BORDER_REPLICATE",
            cv2.BORDER_REFLECT: "BORDER_REFLECT",
            cv2.BORDER_WRAP: "BORDER_WRAP",
            cv2.BORDER_REFLECT_101: "BORDER_REFLECT_101",
        }[border_type]
        ax.set_title(f"{border_type_name}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    cv2.destroyAllWindows()


image = cv2.imread("C:\Data\TUM\POSTECH\CompPhot\Homework\Workspace\ComputationalPhotography\Homework1\images\own_images\Image1.jpeg")
compare_border_types(image, border_size=200)
