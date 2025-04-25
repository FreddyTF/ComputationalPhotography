import typing
from typing import List, Tuple
import numpy as np
import cv2
from feature_detect import detect_features
from feature_match import match_features
from ransac import ransac
from homography import compute_average_homography
from warp import warp_and_stich_images, warp_and_stich_images_2
from matplotlib import pyplot as plt
import glob
import os


def compute_correspondence_score(img1: np.ndarray, img2: np.ndarray, features) -> float:
    # Compute the correspondence score between two images
    # This function will be called in the main function
    # It will take two images and return a score
    features = "sift"
    # Dummy implementation, replace with actual computation
    kp1, des1 = detect_features(img1, mode=features, visualize=False)
    kp2, des2 = detect_features(img2, mode=features, visualize=False)

    matches = match_features(
        img1, kp1, des1, img2, kp2, des2, features=features, visualize=False
    )

    ransac_matches, H = ransac(
        matches, img1, img2, kp1, kp2, threshold=3.0, patch_size=7, iterations=5000
    )

    score = len(ransac_matches)
    return score


def stich_images(img1: np.ndarray, img2: np.ndarray, features) -> np.ndarray:
    # Stitch two images together using homography
    kp1, des1 = detect_features(img1, mode=features, visualize=False)

    kp2, des2 = detect_features(img2, mode=features, visualize=False)

    matches = match_features(
        img1, kp1, des1, img2, kp2, des2, features=features, visualize=False
    )

    ransac_matches, H = ransac(
        matches, img1, img2, kp1, kp2, threshold=3.0, patch_size=7, iterations=5000
    )

    homography = compute_average_homography(
        ransac_matches, kp1, kp2, img1, img2, visualize=True
    )

    print(f"Homography: {homography}")

    result = warp_and_stich_images_2(H, img1, img2)
    return result


def create_panorama(images: List[np.ndarray]) -> np.ndarray:
    # Create a panorama from a list of images
    # This function will be called in the main function
    # It will take a list of images and return a panorama image
    features = "sift"

    if len(images) > 2:
        # set first image as reference
        image = images[0]

        # try all n over k possibilites of iamge for the start
        correspondence_score = 0
        correspondence_image_index_1 = 0
        correspondence_image_index_2 = 1

        if len(images) > 1:
            for i in range(1, len(images)):
                for j in range(i + 1, len(images)):
                    this_score = compute_correspondence_score(
                        image, images[i], features=features
                    )
                    if this_score > correspondence_score:
                        correspondence_score = this_score

                        correspondence_image_index_1 = i
                        correspondence_image_index_2 = j
                        # set this image as correspondence image

        ref_image = images[correspondence_image_index_1]
        correspondence_image = images[correspondence_image_index_2]

        stiched = stich_images(ref_image, correspondence_image, features=features)
        images = [
            img
            for img in images
            if img is not ref_image and img is not correspondence_image
        ]
        images.insert(0, stiched)

    while len(images) > 2:
        # set first image as reference
        ref_image = images[0]

        # try all n over k possibilites of iamge for the start
        correspondence_score = 0
        correspondence_image_index_1 = 1

        for i in range(1, len(images)):
            this_score = compute_correspondence_score(
                ref_image, images[i], features=features
            )
            if this_score > correspondence_score:
                correspondence_score = this_score
                correspondence_image_index_1 = i

                # set this image as correspondence image

        correspondence_image = images[correspondence_image_index_1]

        stiched = stich_images(ref_image, correspondence_image, features=features)
        images = [
            img
            for img in images
            if img is not ref_image and img is not correspondence_image
        ]
        images.insert(0, stiched)

    if len(images) == 2:
        img1 = images[0]
        img2 = images[1]
        images = [stich_images(img1, img2, features=features)]

    return images[0]


def main():
    image_files = glob.glob(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../images/panorama4/*.jpeg")
        )
    )

    images = []

    for image_file in image_files:
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    create_panorama(images)


if __name__ == "__main__":
    main()
