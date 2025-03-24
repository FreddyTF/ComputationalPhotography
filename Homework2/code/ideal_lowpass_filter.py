import cv2
import numpy as np
import typing
import matplotlib.pyplot as plt


class ImageForms:
    def __init__(self, image: np.ndarray,fourier_transform: np.ndarray, lowpass_filter: np.ndarray, filtering_result: np.ndarray, result: np.ndarray):
        self.image = image
        self.fourier_transform = fourier_transform
        self.lowpass_filter = lowpass_filter
        self.filtering_result = filtering_result
        self.result = filtering_result


def ideal_lowpass_filter(image, threshold_value):
    """
    Apply ideal lowpass filter to an image.
    :param image: input image
    :param threshold_value: radius of lowpass filter in frequency domain
    :return: filtered image
    """

    # implement image boundary handling to avoid color bleeding across the image
    pad_size = 10
    image = cv2.copyMakeBorder(
        image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT
    )
    # split RGB image into channels
    b, g, r = cv2.split(image)
    channels = [b, g, r]
    channels_filtered = []

    # create a mask to apply the filter only to the image and not the padding
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (
                np.sqrt((i - mask.shape[0] // 2) ** 2 + (j - mask.shape[1] // 2) ** 2)
                > threshold_value
            ):
                mask[i, j] = 0

    for channel in channels:
        img_f = channel

        # fourier transform
        img_f = np.fft.fft2(img_f)

        # img_f = psf2otf(img_f, channel.shape)

        img_f = np.fft.fftshift(img_f)

        # apply ideal lowpass filter
        # lowest frequency is at the center of the image
        # set all frequencies outside the threshold to zero

        # inverse fourier transform
        img_f = np.fft.ifftshift(img_f)
        channel_filtered = np.real(np.fft.ifft2(img_f))

        # remove padding
        # Crop back to original size to remove padding
        cropped = channel_filtered[pad_size:-pad_size, pad_size:-pad_size]
        channels_filtered.append(cropped)

    return cv2.merge(channels_filtered).astype(np.float32)


def psf2otf(flt, image_shape):
    # pad zeros and shift the center of flt
    flt_top_half = flt.shape[0] // 2
    flt_bottom_half = flt.shape[0] - flt_top_half
    flt_left_half = flt.shape[1] // 2
    flt_right_half = flt.shape[1] - flt_left_half
    flt_padded = np.zeros(image_shape, dtype=flt.dtype)
    # in the top left corner of flt_padded assign flt bottom right corner of flt
    flt_padded[:flt_bottom_half, :flt_right_half] = flt[flt_top_half:, flt_left_half:]
    # in the top right corner of flt_padded assign flt bottom left corner of flt
    flt_padded[:flt_bottom_half, image_shape[1] - flt_left_half :] = flt[
        flt_top_half:, :flt_left_half
    ]
    # in the bottom left corner of flt_padded assign flt top right corner of flt
    flt_padded[image_shape[0] - flt_top_half :, :flt_right_half] = flt[
        :flt_top_half, flt_left_half:
    ]
    # in the bottom right corner of flt_padded assign flt top left corner of flt
    flt_padded[image_shape[0] - flt_top_half :, image_shape[1] - flt_left_half :] = flt[
        :flt_top_half, :flt_left_half
    ]
    return np.fft.fft2(flt_padded)
