import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../code')))

from filter import gaussian_1d
import cv2
import numpy as np


def test_gaussian_1d():
    # Test the function with a sigma of 1 and a size of 3
    true_kernel = np.array([0.27406862, 0.45186276, 0.27406862])
    test_kernel = gaussian_1d(1, 3)

    print(f"{true_kernel = }")
    print(f"{test_kernel = }")

    assert np.allclose(true_kernel, test_kernel)
