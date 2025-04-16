import pytest
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

from main import create_panorama
import cv2
import glob


def test_overlapping_images():
    images = []

    image_files = glob.glob(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../images/panorama1/*.jpeg")
        )
    )
    for file in image_files:
        img = cv2.imread(file)
        if img is not None:
            images.append(img)

    ground_truth = cv2.imread(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../images/panorama1_ground_truth/WhatsApp Image 2025-04-15 at 21.39.33.jpeg",
            )
        )
    )

    result = create_panorama(images)

    assert np.allclose(result, ground_truth, atol=1e-2), (
        "The panorama does not match the ground truth image."
    )
