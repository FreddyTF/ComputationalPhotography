import numpy as np
import cv2
import matplotlib.pyplot as plt
from ideal_lowpass_filter import psf2otf


def gauss(n=11, sigma=1):
    """
    Create a Gaussian filter.
    :param n: size of the filter
    :param sigma: standard deviation of the Gaussian distribution
    :return: Gaussian filter
    """
    # create a 1D Gaussian filter
    r = np.arange(0, n, dtype=np.float32) - (n - 1.0) / 2.0
    r = np.exp(-(r**2.0) / (2.0 * sigma**2))
    return r / np.sum(r)


def gauss2d(shape=(11, 11), sigma=1):
    """
    Create a 2D Gaussian filter.
    :param shape: size of the filter
    :param sigma: standard deviation of the Gaussian distribution
    :return: 2D Gaussian filter
    """
    g1 = gauss(shape[0], sigma).reshape([shape[0], 1])
    g2 = gauss(shape[1], sigma).reshape([1, shape[1]])
    return np.matmul(g1, g2)


def gaussian_lowpass_filter(image, threshold_value, visualize=False) -> np.ndarray:
    pad_size = 10
    image = cv2.copyMakeBorder(
        image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT
    )

    # split RGB image into channels
    b, g, r = cv2.split(image)
    channels = [b, g, r]
    channels_filtered = []
    channels_f = []
    channels_f_filtered = []

    hs = threshold_value  # half of the filter size
    flt = gauss2d((hs * 2 + 1, hs * 2 + 1), hs / 6.0)

    for channel in channels:
        # fourier transform
        channel_f = np.fft.fft2(channel)

        flt_f = psf2otf(flt, channel.shape)

        channel_flt_f = channel_f * flt_f

        # inverse fourier transform
        channel_filtered = np.real(np.fft.ifft2(channel_flt_f))

        cropped = channel_filtered[pad_size:-pad_size, pad_size:-pad_size]

        # remove padding
        # Crop back to original size to remove padding
        channels_filtered.append(cropped)

        channels_f.append(np.fft.fftshift(np.log(np.abs(channel_f) + 1)))
        channels_f_filtered.append(np.fft.fftshift(np.log(np.abs(channel_flt_f) + 1)))

    lowpass_image = cv2.merge(channels_filtered).astype(np.float32)

    if visualize:
        image_f = cv2.merge(channels_f).astype(np.float32)
        image_f = cv2.normalize(image_f, None, 0, 1, cv2.NORM_MINMAX)
        image_f_lowpass = cv2.merge(channels_f_filtered).astype(np.float32)
        image_f_lowpass = cv2.normalize(image_f_lowpass, None, 0, 1, cv2.NORM_MINMAX)
        plot_task_gaussian_lowpass(
            image, lowpass_image, channels_f, channels_f_filtered
        )

    return lowpass_image


def plot_task_gaussian_lowpass(image, lowpass_image, image_f, image_f_lowpass):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image in Spatial Domain")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(image_f, cmap="gray", vmin=0, vmax=1)
    axs[0, 1].set_title("Original Image in Frequency Domain")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(cv2.cvtColor(lowpass_image, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Gaussian Lowpass Image in Spatial Domain")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(image_f_lowpass, cmap="gray", vmin=0, vmax=1)
    axs[1, 1].set_title("Gaussian Lowpass Image in Frequency Domain")
    axs[1, 1].axis("off")

    plt.show()
