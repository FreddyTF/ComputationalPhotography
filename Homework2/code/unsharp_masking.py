import cv2
import numpy as np
from gaussian_lowpass_filter import gauss, gauss2d


def unsharp_masking(image: np.ndarray, domain: str) -> np.ndarray:
    """
    Apply unsharp masking to an image.
    """
    pad_size = 10
    image = cv2.copyMakeBorder(
        image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT
    )

    #  split RGB image into channels
    b, g, r = cv2.split(image)
    channels = [b, g, r]
    channels_filtered = []

    for channel in channels:
        if domain == "spatial":
            channel = unsharp_masking_spatial(channel)
        elif domain == "frequency":
            channel = unsharp_masking_frequency(channel)
        else:
            raise ValueError("Invalid domain parameter.")
        channels_filtered.append(channel)

    return cv2.merge(channels_filtered).astype(np.float32)


def unsharp_masking_spatial(image: np.ndarray) -> np.ndarray:
    """ "
    Apply unsharp masking to an image in spatial domain."
    """

    # Y = X + alpha (X - G conv X)
    # result = image + alpha ( X - G conv X)
    alpha = 3.0
    kernel = gauss2d((5, 5), 1)

    GvonvX = cv2.filter2D(image, -1, kernel)

    result = image + alpha * (image - GvonvX)

    return result


def unsharp_masking_frequency(image: np.ndarray) -> np.ndarray:
    """ """
    # Y = X + alpha (X - G conv X)
    # result = image + alpha ( X - G conv X)

    alpha = 0.5
    kernel = gauss2d((5, 5), 1)

    # convert image to frequency domain
    image_f = np.fft.fft2(image)

    # convert kernel to frequency domain
    kernel_f = np.fft.fft2(kernel, s=image.shape)

    # apply kernel to image
    GvonvX_f = image_f * kernel_f

    result_f = image_f + alpha * (image_f - GvonvX_f)

    # convert result to spatial domain
    result = np.real(np.fft.ifft2(result_f))

    return result
