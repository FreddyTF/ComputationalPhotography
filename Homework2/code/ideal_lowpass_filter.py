import cv2
import numpy as np
import typing
import matplotlib.pyplot as plt


def ideal_lowpass_filter(
    image: np.ndarray, threshold_value: int, visualize=False
) -> np.ndarray:
    """
    Apply ideal lowpass filter to an image.
    :param image: input image
    :param threshold_value: radius of lowpass filter in frequency domain
    :return: filtered image

    # f indicates the fourier transform of the image
    # lowpass for filtered image
    """

    # implement image boundary handling to avoid color bleeding across the image
    pad_size = threshold_value  # padding size
    image = cv2.copyMakeBorder(
        image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT
    )
    # split RGB image into channels
    b, g, r = cv2.split(image)
    channels = [b, g, r]
    channels_cropped_filtered = []
    channels_f = []
    channels_f_filtered = []

    # create a mask to apply the filter only to the image and not the padding
    mask = np.ones(image.shape[:2], dtype=np.uint8)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (
                np.sqrt((i - mask.shape[0] // 2) ** 2 + (j - mask.shape[1] // 2) ** 2)
                > threshold_value
            ):
                mask[i, j] = 0

    for channel in channels:
        # fourier transform
        channel_f = np.fft.fft2(channel)  # for reasons of simplicity now called

        channel_f_shifted = np.fft.fftshift(
            channel_f
        )  # shift the center of the image to the center of the fourier transform

        # apply ideal lowpass filter
        # lowest frequency is at the center of the image
        # set all frequencies outside the threshold to zero
        channel_f_shifted_lowpass = channel_f_shifted * mask

        # inverse fourier transform
        channel_f_lowpass = np.fft.ifftshift(
            channel_f_shifted_lowpass
        )  # reverse the shift
        channel_lowpass = np.real(
            np.fft.ifft2(channel_f_lowpass)
        )  # reverse the fourier transform

        # remove padding
        # Crop back to original size to remove padding
        cropped_channel = channel_lowpass[pad_size:-pad_size, pad_size:-pad_size]
        channels_cropped_filtered.append(cropped_channel)

        channels_f.append(np.fft.fftshift(np.log(np.abs(channel_f) + 1)))

        channels_f_filtered.append(
            np.fft.fftshift(np.log(np.abs(channel_f_lowpass) + 1))
        )

    lowpass_image = cv2.merge(channels_cropped_filtered).astype(np.float32)

    if visualize:
        image_f = cv2.merge(channels_f).astype(np.float32)
        image_f = cv2.normalize(image_f, None, 0, 1, cv2.NORM_MINMAX)
        image_f_lowpass = cv2.merge(channels_f_filtered).astype(np.float32)
        image_f_lowpass = cv2.normalize(image_f_lowpass, None, 0, 1, cv2.NORM_MINMAX)

        plot_task_ideal_lowpass(image, lowpass_image, image_f, image_f_lowpass)

    return lowpass_image


def plot_task_ideal_lowpass(image, lowpass_image, image_f, image_f_lowpass):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image in Spatial Domain")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(image_f, cmap="gray", vmin=0, vmax=1)
    axs[0, 1].set_title("Original Image in Frequency Domain")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(cv2.cvtColor(lowpass_image, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Ideal Lowpass Image in Spatial Domain")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(image_f_lowpass, cmap="gray", vmin=0, vmax=1)
    axs[1, 1].set_title("Ideal Lowpass Image in Frequency Domain")
    axs[1, 1].axis("off")

    plt.show()


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
