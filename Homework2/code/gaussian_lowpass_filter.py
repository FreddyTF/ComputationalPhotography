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


def gaussian_lowpass_filter(image, threshold_value) -> np.ndarray:
    pad_size = 10
    image = cv2.copyMakeBorder(
        image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT
    )

    # split RGB image into channels
    b, g, r = cv2.split(image)
    channels = [b, g, r]
    channels_filtered = []

    hs = threshold_value  # half of the filter size
    flt = gauss2d((hs * 2 + 1, hs * 2 + 1), hs / 6.0)

    for channel in channels:
        img_f = channel

        # fourier transform
        img_f = np.fft.fft2(img_f)

        flt_f = psf2otf(flt, channel.shape)

        img_flt_f = img_f * flt_f

        # inverse fourier transform
        channel_filtered = np.real(np.fft.ifft2(img_flt_f))

        cropped = channel_filtered[pad_size:-pad_size, pad_size:-pad_size]

        # remove padding
        # Crop back to original size to remove padding
        channels_filtered.append(cropped)

    return cv2.merge(channels_filtered).astype(np.float32)
