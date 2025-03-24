import cv2
import numpy as np
import matplotlib.pyplot as plt

from ideal_lowpass_filter import ideal_lowpass_filter
from gaussian_lowpass_filter import gaussian_lowpass_filter
from unsharp_masking import unsharp_masking


def main():
    ideal_lowpass_filter_use = True
    gaussian_lowpass_filter_use = False
    unsharp_masking_use_spatial = False
    unsharp_masking_use_frequency = False

    # ideal lowpass filter

    image = (
        cv2.imread(
            # r"images\own_images\dji_mimo_20250321_113618_0_1742555647725_photo.jpg",
            r"images\own_images\dji_fly_20250321_124130_671_1742555378877_photo.jpg",
            # r"images\own_images\templemap.jpg",
            # r"images/color3.jpg",
            cv2.IMREAD_COLOR,
        ).astype(np.float32)
        / 255.0
    )

    threshold_value = 50  # radius of lowpass filter in frequency domain

    if ideal_lowpass_filter_use:
        output = ideal_lowpass_filter(
            image, threshold_value, visualize=True
        )  # filtered result
    elif gaussian_lowpass_filter_use:
        output = gaussian_lowpass_filter(image, threshold_value)
    elif unsharp_masking_use_spatial:
        output = unsharp_masking(image, "spatial")  # filtered result
    elif unsharp_masking_use_frequency:
        output = unsharp_masking(image, "frequency")

    # Display the original and filtered images
    plt.figure(figsize=(10, 5))

    # Original image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.imshow(image, vmin=0, vmax=1)
    plt.title("Original Image")
    plt.axis("off")

    # Filtered image
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 2)
    plt.imshow(output, vmin=0, vmax=1)
    plt.title("Filtered Image")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
