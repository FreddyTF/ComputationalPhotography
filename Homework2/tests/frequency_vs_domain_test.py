import pytest
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

from unsharp_masking import unsharp_masking


def test_frequency_vs_domain():
    # Test the frequency_vs_domain function with a sample image and threshold value
    image = np.random.rand(1000, 1000, 3)  # Sample random image
    alpha = 1.5  # Sample alpha value for unsharp masking
    kernel_size = 5  # Sample kernel size for unsharp masking
    kernel_sigma = 1.0  # Sample kernel sigma for unsharp masking

    image_domain = unsharp_masking(
        image=image,
        domain="spatial",
        alpha=alpha,
        kernel_size=kernel_size,
        kernel_sigma=kernel_sigma,
    )

    image_frequency = unsharp_masking(
        image=image,
        domain="frequency",
        alpha=alpha,
        kernel_size=kernel_size,
        kernel_sigma=kernel_sigma,
    )

    assert np.allclose(image_domain, image_frequency, atol=1e-5), (
        "The images in frequency and spatial domain do not match!"
    )
