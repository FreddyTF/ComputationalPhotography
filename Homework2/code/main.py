import cv2
import numpy as np
import matplotlib.pyplot as plt

from ideal_lowpass_filter import ideal_lowpass_filter
from gaussian_lowpass_filter import (
    gaussian_lowpass_filter,
    gaussian_lowpass_filter_variable_border_size,
)
from unsharp_masking import (
    unsharp_masking,
    image_loader_compare_unmasking_spatial_and_frequency,
    image_loader_compare_parameters,
    compare_run_time,
)


def main():
    ideal_lowpass_filter_use = False  # True
    gaussian_lowpass_filter_use = False
    unsharp_masking_use_spatial = False
    unsharp_masking_use_frequency = True

    # ideal lowpass filter

    image = (
        cv2.imread(
            # r"images\own_images\dji_mimo_20250321_113618_0_1742555647725_photo.jpg",
            # r"images\own_images\dji_fly_20250321_124130_671_1742555378877_photo.jpg",
            # r"images\own_images\templemap.jpg",
            r"images/color3.jpg",
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
        output = gaussian_lowpass_filter(image, threshold_value, visualize=True)
    elif unsharp_masking_use_spatial:
        kernel_size = 3
        kernel_sigma = 1
        alpha = 0.5
        output = unsharp_masking(
            image=image,
            domain="spatial",
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            alpha=alpha,
        )  # filtered result
    elif unsharp_masking_use_frequency:
        kernel_size = 3
        kernel_sigma = 1
        alpha = 0.5
        output = unsharp_masking(
            image,
            domain="frequency",
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            alpha=alpha,
        )

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


def show_different_ideal_lowpass_filter_results():
    image = (
        cv2.imread(
            r"images\own_images\dji_mimo_20250321_113618_0_1742555647725_photo.jpg",
            # r"images\own_images\dji_fly_20250321_124130_671_1742555378877_photo.jpg",
            # r"images\own_images\templemap.jpg",
            # r"images/color3.jpg",
            cv2.IMREAD_COLOR,
        ).astype(np.float32)
        / 255.0
    )

    thresholds = [5, 20, 50, 100, 200]

    results = []

    for threshold in thresholds:
        result = ideal_lowpass_filter(image, threshold, visualize=False)
        results.append(result)

    plt.figure(figsize=(8, 6))

    plt.subplot(3, 2, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb, vmin=0, vmax=1)
    plt.title("Original Image")
    plt.axis("off")

    for i, result in enumerate(results):
        plt.subplot(3, 2, i + 2)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(result_rgb, vmin=0, vmax=1)
        plt.title(f"Filtered Image (Threshold Size {thresholds[i]})")
        plt.axis("off")

    plt.savefig(
        "results\ideal_lowpass_filter\IdealLowPassDifferenThresholdsEntrance.png",
        dpi=250,
    )
    plt.show()


def show_different_gaussian_lowpass_filter_results():
    image = (
        cv2.imread(
            r"images\own_images\dji_mimo_20250321_113618_0_1742555647725_photo.jpg",
            # r"images\own_images\dji_fly_20250321_124130_671_1742555378877_photo.jpg",
            # r"images\own_images\templemap.jpg",
            # r"images/color3.jpg",
            cv2.IMREAD_COLOR,
        ).astype(np.float32)
        / 255.0
    )

    thresholds = [5, 20, 35, 50, 100]

    results = []

    for threshold in thresholds:
        result = gaussian_lowpass_filter(image, threshold, visualize=False)
        results.append(result)

    plt.figure(figsize=(7, 6))

    plt.subplot(3, 2, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb, vmin=0, vmax=1)
    plt.title("Original Image")
    plt.axis("off")

    for i, result in enumerate(results):
        plt.subplot(3, 2, i + 2)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(result_rgb, vmin=0, vmax=1)
        plt.title(f"Filtered Image (Threshold {thresholds[i]})")
        plt.axis("off")

    plt.show()


def compare_ideal_and_gaussian_lowpass_filter():
    image_paths = [
        r"images\own_images\dji_mimo_20250321_113618_0_1742555647725_photo.jpg",
        r"images\own_images\dji_fly_20250321_124130_671_1742555378877_photo.jpg",
        r"images/color3.jpg",
    ]

    for i, image_path in enumerate(image_paths):
        image = (
            cv2.imread(
                image_path,
                cv2.IMREAD_COLOR,
            ).astype(np.float32)
            / 255.0
        )

        threshold_value = 50  # radius of lowpass filter in frequency domain

        output_ideal = ideal_lowpass_filter(image, threshold_value, visualize=False)
        output_gaussian = gaussian_lowpass_filter(
            image, threshold_value, visualize=False
        )

        plt.figure(figsize=(10, 5))
        # Original image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb, vmin=0, vmax=1)
        plt.title("Original Image")
        plt.axis("off")

        # Ideal lowpass filter result
        output_ideal_rgb = cv2.cvtColor(output_ideal, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, 2)
        plt.imshow(output_ideal_rgb, vmin=0, vmax=1)
        plt.title("Ideal Lowpass Filter")
        plt.axis("off")

        # Gaussian lowpass filter result
        output_gaussian_rgb = cv2.cvtColor(output_gaussian, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, 3)
        plt.imshow(output_gaussian_rgb, vmin=0, vmax=1)
        plt.title("Gaussian Lowpass Filter")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"results/compare_ideal_gaussian_lp_filter/image{i}.png", dpi=250)
        plt.close()


def compare_border_padding_for_gaussian():
    image = (
        cv2.imread(
            # r"images\own_images\dji_mimo_20250321_113618_0_1742555647725_photo.jpg",
            # r"images\own_images\dji_fly_20250321_124130_671_1742555378877_photo.jpg",
            # r"images\own_images\templemap.jpg",
            r"images/color3.jpg",
            cv2.IMREAD_COLOR,
        ).astype(np.float32)
        / 255.0
    )

    threshold = 30
    border_sizes = [0, 30, 60, 80, 100]
    results = []

    for border_size in border_sizes:
        result = gaussian_lowpass_filter_variable_border_size(
            image, threshold, visualize=False, border_size=border_size
        )
        results.append(result)

    plt.figure(figsize=(7, 6))

    plt.subplot(3, 2, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb, vmin=0, vmax=1)
    plt.title("Original Image")
    plt.axis("off")

    for i, result in enumerate(results):
        plt.subplot(3, 2, i + 2)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(result_rgb, vmin=0, vmax=1)
        plt.title(f"Filtered Image (Bordersize {border_sizes[i]})")
        plt.axis("off")

    plt.savefig(
        "results/gaussian_lowpass_filter_border_padding_comparison.png", dpi=250
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    # main()
    show_different_ideal_lowpass_filter_results()
    # show_different_gaussian_lowpass_filter_results()
    # image_loader_compare_unmasking_spatial_and_frequency()
    # image_loader_compare_parameters()
    # compare_run_time()
    # compare_ideal_and_gaussian_lowpass_filter()
    # compare_border_padding_for_gaussian()
    pass
