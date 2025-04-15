import cv2
import numpy as np
from gaussian_lowpass_filter import gauss2d
import matplotlib.pyplot as plt
import time
from ideal_lowpass_filter import psf2otf


def unsharp_masking(
    image: np.ndarray, domain: str, kernel_size: int, kernel_sigma: float, alpha: float
) -> np.ndarray:
    """
    Apply unsharp masking to an image.
    """
    pad_size = kernel_size // 2  # padding size
    image = cv2.copyMakeBorder(
        image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT
    )

    #  split RGB image into channels
    b, g, r = cv2.split(image)
    channels = [b, g, r]
    channels_filtered = []

    for channel in channels:
        if domain == "spatial":
            channel = unsharp_masking_spatial(
                image=channel,
                alpha=alpha,
                kernel_size=kernel_size,
                kernel_sigma=kernel_sigma,
            )
        elif domain == "frequency":
            channel = unsharp_masking_frequency(
                image=channel,
                alpha=alpha,
                kernel_size=kernel_size,
                kernel_sigma=kernel_sigma,
            )
        else:
            raise ValueError("Invalid domain parameter.")
        channels_filtered.append(channel)

    sharped_image = cv2.merge(channels_filtered).astype(np.float32)
    image_cropped = sharped_image[pad_size:-pad_size, pad_size:-pad_size]

    return image_cropped


def unsharp_masking_spatial(
    image: np.ndarray, alpha: float, kernel_size: int, kernel_sigma: float
) -> np.ndarray:
    """ "
    Apply unsharp masking to an image in spatial domain."
    """

    # Y = X + alpha (X - G conv X)
    # result = image + alpha ( X - G conv X)
    kernel = gauss2d((kernel_size, kernel_size), kernel_sigma)

    GvonvX = cv2.filter2D(image, -1, kernel)

    result = image + alpha * (image - GvonvX)

    return result


def unsharp_masking_frequency(
    image: np.ndarray, alpha: float, kernel_size: int, kernel_sigma: float
) -> np.ndarray:
    """ """
    # Y = X + alpha (X - G conv X)
    # result = image + alpha ( X - G conv X)
    print("Frequency Domain")
    kernel = gauss2d((kernel_size, kernel_size), sigma=kernel_sigma)

    # convert image to frequency domain
    image_f = np.fft.fft2(image)

    # convert kernel to frequency domain
    kernel_f = psf2otf(kernel, image.shape)

    # kernel_f = np.fft.fft2(kernel, s=image.shape)

    # apply kernel to image
    GvonvX_f = image_f * kernel_f

    result_f = image_f + alpha * (image_f - GvonvX_f)

    # convert result to spatial domain
    result = np.real(np.fft.ifft2(result_f))

    return result


def compare_unmasking_spatial_and_frequency(
    image: np.ndarray, counter: int, alpha: float, kernel_size: int, kernel_sigma: float
):
    """
    Compare the results of unsharp masking in spatial and frequency domain.
    """

    # spatial domain
    output_spatial = unsharp_masking(
        image=image,
        domain="spatial",
        alpha=alpha,
        kernel_size=kernel_size,
        kernel_sigma=kernel_sigma,
    )  # filtered result

    # frequency domain
    output_frequency = unsharp_masking(
        image=image,
        domain="frequency",
        alpha=alpha,
        kernel_size=kernel_size,
        kernel_sigma=kernel_sigma,
    )

    # Display the original and filtered images
    plt.figure(figsize=(10, 5))

    # Original image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 1)
    plt.imshow(image, vmin=0, vmax=1)
    plt.title("Original Image")
    plt.axis("off")

    # Filtered image
    output_spatial = cv2.cvtColor(output_spatial, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 2)
    plt.imshow(output_spatial, vmin=0, vmax=1)
    plt.title(
        f"Filtered Image (Spatial) \n {kernel_size = } \n {kernel_sigma = } \n {alpha = }"
    )
    plt.axis("off")

    # Filtered image
    output_frequency = cv2.cvtColor(output_frequency, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 3)
    plt.imshow(output_frequency, vmin=0, vmax=1)
    plt.title(
        f"Filtered Image (Frequency) \n {kernel_size = } \n {kernel_sigma = }  \n {alpha = }"
    )
    plt.axis("off")

    plt.savefig(
        r"results\unsharp_masking\compare_spatial_and_frequency_domain\image"
        + str(counter)
        + ".png",
        dpi=300,  # high resolution
    )


def image_loader_compare_unmasking_spatial_and_frequency():
    image_path = [
        r"images\own_images\dji_mimo_20250321_113618_0_1742555647725_photo.jpg",
        r"images\own_images\dji_fly_20250321_124130_671_1742555378877_photo.jpg",
        r"images\own_images\templemap.jpg",
        r"images\color3.jpg",
    ]

    kernel_size = [3, 5, 7, 11]
    alpha = [10, 4, 8, 16]
    kernel_sigma = [1, 2, 4, 8]

    for counter, image_path in enumerate(image_path):
        image = (
            cv2.imread(
                image_path,
                cv2.IMREAD_COLOR,
            ).astype(np.float32)
            / 255.0
        )
        compare_unmasking_spatial_and_frequency(
            image=image,
            counter=counter,
            alpha=alpha[0],
            kernel_size=kernel_size[0],
            kernel_sigma=kernel_sigma[0],
        )


def image_loader_compare_parameters():
    image_path = [
        # r"images\own_images\dji_mimo_20250321_113618_0_1742555647725_photo.jpg",
        r"images\own_images\dji_fly_20250321_124130_671_1742555378877_photo.jpg",
        # r"images\own_images\templemap.jpg",
        r"images\color3.jpg",
    ]

    kernel_size = [3, 7, 11, 15]
    alpha = [0.1, 0.5, 3, 6]
    kernel_sigma = [0.5, 1, 2, 4]

    for counter, image_path in enumerate(image_path):
        image = (
            cv2.imread(
                image_path,
                cv2.IMREAD_COLOR,
            ).astype(np.float32)
            / 255.0
        )

        fig, axes = plt.subplots(len(kernel_size), len(kernel_sigma), figsize=(15, 15))
        fig.suptitle(f"Unsharp Masking with Fixed Alpha = {alpha[3]}", fontsize=16)
        # fixed alpha
        for i, ks in enumerate(kernel_size):
            for j, sigma in enumerate(kernel_sigma):
                output = unsharp_masking(
                    image=image,
                    domain="frequency",
                    alpha=alpha[3],
                    kernel_size=ks,
                    kernel_sigma=sigma,
                )
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                ax = axes[i, j]
                ax.imshow(output, vmin=0, vmax=1)
                ax.set_title(f"ks={ks}, sigma={sigma}")
                ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            r"results\unsharp_masking\compare_parameters\image"
            + str(counter)
            + "_fixed_alpha.png",
            dpi=100,  # high resolution
        )
        plt.close(fig)

        # fixed kernel size
        fig, axes = plt.subplots(len(alpha), len(kernel_sigma), figsize=(15, 15))
        fig.suptitle(
            f"Unsharp Masking with Fixed Kernel Size = {kernel_size[1]}", fontsize=16
        )

        for i, a in enumerate(alpha):
            for j, sigma in enumerate(kernel_sigma):
                output = unsharp_masking(
                    image=image,
                    domain="frequency",
                    alpha=a,
                    kernel_size=kernel_size[1],
                    kernel_sigma=sigma,
                )
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                ax = axes[i, j]
                ax.imshow(output, vmin=0, vmax=1)
                ax.set_title(f"alpha={a}, sigma={sigma}")
                ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            r"results\unsharp_masking\compare_parameters\image"
            + str(counter)
            + "_fixed_kernel_size.png",
            dpi=100,  # high resolution
        )
        plt.close(fig)

        # fixed kernel sigma

        fig, axes = plt.subplots(len(alpha), len(kernel_size), figsize=(15, 15))
        fig.suptitle(
            f"Unsharp Masking with Fixed Kernel Sigma = {kernel_sigma[1]}", fontsize=16
        )

        for i, a in enumerate(alpha):
            for j, ks in enumerate(kernel_size):
                output = unsharp_masking(
                    image=image,
                    domain="frequency",
                    alpha=a,
                    kernel_size=ks,
                    kernel_sigma=kernel_sigma[1],
                )
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                ax = axes[i, j]
                ax.imshow(output, vmin=0, vmax=1)
                ax.set_title(f"alpha={a}, ks={ks}")
                ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            r"results\unsharp_masking\compare_parameters\image"
            + str(counter)
            + "_fixed_kernel_sigma.png",
            dpi=100,  # high resolution
        )
        plt.close(fig)


def compare_run_time():
    image_path = [
        r"images\own_images\dji_mimo_20250321_113618_0_1742555647725_photo.jpg",
        r"images\color3.jpg",
    ]

    kernel_size = [3, 31, 61, 101, 151]
    alpha = [0.0, 1, 50]
    kernel_sigma = [0.01, 1, 100]

    for counter, image_path in enumerate(image_path):
        image = (
            cv2.imread(
                image_path,
                cv2.IMREAD_COLOR,
            ).astype(np.float32)
            / 255.0
        )

        # kernel size
        run_time_frequency = []
        run_time_spatial = []
        for ks in kernel_size:
            start_time = time.time()
            output = unsharp_masking(
                image=image,
                domain="frequency",
                alpha=alpha[0],
                kernel_size=ks,
                kernel_sigma=kernel_sigma[0],
            )
            end_time = time.time()
            run_time_frequency.append(end_time - start_time)

            start_time = time.time()
            output = unsharp_masking(
                image=image,
                domain="spatial",
                alpha=alpha[0],
                kernel_size=ks,
                kernel_sigma=kernel_sigma[0],
            )
            end_time = time.time()
            run_time_spatial.append(end_time - start_time)

        plt.figure(figsize=(8, 6))
        plt.plot(kernel_size, run_time_frequency, label="Frequency Domain")
        plt.plot(kernel_size, run_time_spatial, label="Spatial Domain")
        plt.xlabel("Kernel Size")
        plt.ylabel("Run Time (s)")
        plt.title(
            f"Run Time Comparison (Image Size: {image.shape[1]}x{image.shape[0]})"
        )
        plt.legend()
        plt.grid()
        plt.savefig(
            r"results\unsharp_masking\compare_runtime\image_kernel_size_"
            + str(counter)
            + ".png",
            dpi=300,  # high resolution
        )
        plt.close()

        # kernel sigma
        run_time_frequency = []
        run_time_spatial = []
        for sigma in kernel_sigma:
            start_time = time.time()
            output = unsharp_masking(
                image=image,
                domain="frequency",
                alpha=alpha[0],
                kernel_size=kernel_size[0],
                kernel_sigma=sigma,
            )
            end_time = time.time()
            run_time_frequency.append(end_time - start_time)

            start_time = time.time()
            output = unsharp_masking(
                image=image,
                domain="spatial",
                alpha=alpha[0],
                kernel_size=kernel_size[0],
                kernel_sigma=sigma,
            )
            end_time = time.time()
            run_time_spatial.append(end_time - start_time)

        plt.figure(figsize=(8, 6))
        plt.plot(kernel_sigma, run_time_frequency, label="Frequency Domain")
        plt.plot(kernel_sigma, run_time_spatial, label="Spatial Domain")
        plt.xlabel("Kernel Sigma")
        plt.ylabel("Run Time (s)")
        plt.title(
            f"Run Time Comparison (Image Size: {image.shape[1]}x{image.shape[0]})"
        )
        plt.legend()
        plt.grid()
        plt.savefig(
            r"results\unsharp_masking\compare_runtime\image_kernel_sigma_"
            + str(counter)
            + ".png",
            dpi=300,  # high resolution
        )
        plt.close()

        # kernel alpha
        run_time_frequency = []
        run_time_spatial = []
        for a in alpha:
            start_time = time.time()
            output = unsharp_masking(
                image=image,
                domain="frequency",
                alpha=a,
                kernel_size=kernel_size[0],
                kernel_sigma=kernel_sigma[0],
            )
            end_time = time.time()
            run_time_frequency.append(end_time - start_time)

            start_time = time.time()
            output = unsharp_masking(
                image=image,
                domain="spatial",
                alpha=a,
                kernel_size=kernel_size[0],
                kernel_sigma=kernel_sigma[0],
            )
            end_time = time.time()
            run_time_spatial.append(end_time - start_time)

        plt.figure(figsize=(8, 6))
        plt.plot(kernel_sigma, run_time_frequency, label="Frequency Domain")
        plt.plot(kernel_sigma, run_time_spatial, label="Spatial Domain")
        plt.xlabel("Kernel Alpha")
        plt.ylabel("Run Time (s)")
        plt.title(
            f"Run Time Comparison (Image Size: {image.shape[1]}x{image.shape[0]})"
        )
        plt.legend()
        plt.grid()
        plt.savefig(
            r"results\unsharp_masking\compare_runtime\image_alpha_"
            + str(counter)
            + ".png",
            dpi=300,  # high resolution
        )
        plt.close()

    pass
