import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

from filter import gaussian_1d, gaussian_2d, filterGaussian, gaussian_2d_direct
import cv2
import numpy as np

from scipy.signal import gaussian as scipy_gaussian_1d


def test_gaussian_1d():
    # Test the function with a sigma of 1 and a size of 3
    # Params
    size = 3
    sigma = 1

    # Setup
    true_kernel = scipy_gaussian_1d(size, std=sigma)

    # Normalize so that the sum is 1
    true_kernel /= true_kernel.sum()

    test_kernel = gaussian_1d(sigma=sigma, size=size)

    print(f"{true_kernel = }")
    print(f"{test_kernel = }")

    assert np.allclose(true_kernel, test_kernel)


def test_gaussian_1d_2():
    # Test the function with a sigma of 2 and a size of 5

    # Params
    size = 5
    sigma = 2

    # Setup
    true_kernel = scipy_gaussian_1d(size, std=sigma)

    # Normalize so that the sum is 1
    true_kernel /= true_kernel.sum()
    test_kernel = gaussian_1d(sigma=sigma, size=size)

    print(f"{true_kernel = }")
    print(f"{test_kernel = }")

    assert np.allclose(true_kernel, test_kernel)


def test_gaussian_1d_3():
    # Test the function with a sigma of 2 and a size of 5

    # Params
    size = 3
    sigma = 3

    # Setup
    true_kernel = scipy_gaussian_1d(size, std=sigma)

    # Normalize so that the sum is 1
    true_kernel /= true_kernel.sum()
    test_kernel = gaussian_1d(sigma=sigma, size=size)

    print(f"{true_kernel = }")
    print(f"{test_kernel = }")

    assert np.allclose(true_kernel, test_kernel)


def test_gaussian_2d():
    # Params
    size = 3
    sigma = 1

    # Setup
    true_kernel = scipy_gaussian_1d(size, std=sigma)
    true_kernel_2d = np.outer(true_kernel, true_kernel)

    # Normalize
    true_kernel_2d = true_kernel_2d / true_kernel_2d.sum()

    test_kernel = gaussian_2d(sigma=sigma, size=size)

    print(f"{true_kernel_2d = }")
    print(f"{true_kernel_2d.sum() = }")
    print(f"{test_kernel = }")
    print(f"{test_kernel.sum() = }")
    assert np.allclose(true_kernel_2d, test_kernel)


def test_gaussian_2d_2():
    # Params
    size = 5
    sigma = 2

    # Setup
    true_kernel = scipy_gaussian_1d(size, std=sigma)
    true_kernel_2d = np.outer(true_kernel, true_kernel)

    # Normalize
    true_kernel_2d = true_kernel_2d / true_kernel_2d.sum()

    test_kernel = gaussian_2d(sigma=sigma, size=size)

    assert np.allclose(true_kernel_2d, test_kernel)


def test_gaussian_2d_direct():
    # Params
    size = 3
    sigma = 2

    # Setup
    true_kernel = scipy_gaussian_1d(size, std=sigma)
    true_kernel_2d = np.outer(true_kernel, true_kernel)
    true_kernel_2d = true_kernel_2d / true_kernel_2d.sum()

    test_kernel_2d = gaussian_2d_direct(sigma=sigma, size=size)

    assert np.allclose(true_kernel_2d, test_kernel_2d)


def test_gaussian_2d_direct_2():
    # Params
    size = 5
    sigma = 3

    # Setup
    true_kernel = scipy_gaussian_1d(size, std=sigma)
    true_kernel_2d = np.outer(true_kernel, true_kernel)
    true_kernel_2d = true_kernel_2d / true_kernel_2d.sum()

    test_kernel_2d = gaussian_2d_direct(sigma=sigma, size=size)

    assert np.allclose(true_kernel_2d, test_kernel_2d)


def test_gaussian_2d_direct_3():
    # Params
    size = 7
    sigma = 1

    # Setup
    true_kernel = scipy_gaussian_1d(size, std=sigma)
    true_kernel_2d = np.outer(true_kernel, true_kernel)
    true_kernel_2d = true_kernel_2d / true_kernel_2d.sum()

    test_kernel_2d = gaussian_2d_direct(sigma=sigma, size=size)

    assert np.allclose(true_kernel_2d, test_kernel_2d)


def test_filter_gaussian_separable_identical():
    """
    Test the filterGaussian function
    """
    # Create a random colored image
    image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    # Apply the Gaussian filter
    filtered_image = filterGaussian(
        image=image,
        kernel_size=3,
        kernel_sigma=0.001,
        border_type=cv2.BORDER_CONSTANT,
        separable=True,
    )

    # Check the shape of the filtered image
    assert filtered_image.shape == image.shape

    # Check if the filtered image is still within the valid range
    assert np.allclose(filtered_image, image)


def test_filter_gaussian_non_separable_identical():
    """
    Test the filterGaussian function
    """
    # Create a random colored image
    image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    # Apply the Gaussian filter
    filtered_image = filterGaussian(
        image=image,
        kernel_size=3,
        kernel_sigma=0.01,
        border_type=cv2.BORDER_CONSTANT,
        separable=False,
    )

    # Check the shape of the filtered image
    assert filtered_image.shape == image.shape

    # Check if the filtered image is still within the valid range
    assert np.allclose(filtered_image, image)


def test_filter_gaussian_non_separable_non_identical():
    """
    Test the filterGaussian function
    """
    # Create a random colored image
    image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    # Apply the Gaussian filter
    filtered_image = filterGaussian(
        image=image,
        kernel_size=3,
        kernel_sigma=1,
        border_type=cv2.BORDER_CONSTANT,
        separable=False,
    )

    # Check the shape of the filtered image
    assert filtered_image.shape == image.shape

    # Check if the filtered image is still within the valid range
    assert not np.allclose(filtered_image, image)


def test_filter_gaussian_separable_non_identical():
    """
    Test the filterGaussian function
    """
    # Create a random colored image
    image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    # Apply the Gaussian filter
    filtered_image = filterGaussian(
        image=image,
        kernel_size=3,
        kernel_sigma=1,
        border_type=cv2.BORDER_CONSTANT,
        separable=True,
    )

    # Check the shape of the filtered image
    assert filtered_image.shape == image.shape

    # Check if the filtered image is still within the valid range
    assert not np.allclose(filtered_image, image)
