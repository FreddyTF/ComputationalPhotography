import cv2
import numpy as np
import typing
from homography import apply_homography, invert_homography
from matplotlib import pyplot as plt


def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32(
        [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]
    ).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(
        -1, 1, 2
    )

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array(
        [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
    )

    output_img = cv2.warpPerspective(
        img2, H_translation.dot(H), (x_max - x_min, y_max - y_min)
    )
    output_img[
        translation_dist[1] : rows1 + translation_dist[1],
        translation_dist[0] : cols1 + translation_dist[0],
    ] = img1

    return output_img


def warp_images_inverse(
    homography: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray,
    visualise: bool = False,
) -> np.ndarray:
    """
    Warp image2 to the perspective of image1 using the provided homography matrix.

    Parameters:
        homography (np.ndarray): The homography matrix to warp image2.
        image1 (np.ndarray): The first image (base image).
        image2 (np.ndarray): The second image to be warped.

    Returns:
        np.ndarray: The warped panorama image.
    """

    # Get the dimensions of the images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Define the corners of the images
    corners_img1 = np.array(
        [[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32
    ).reshape(-1, 1, 2)
    corners_img2 = np.array(
        [[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32
    ).reshape(-1, 1, 2)

    inverted_homography = invert_homography(homography)
    # Warp the corners of image2 using the homography matrix
    warped_corners = cv2.perspectiveTransform(corners_img2, inverted_homography)

    # Combine the corners of both images to find the bounding box
    all_corners = np.concatenate((warped_corners, corners_img1), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Create the panorama canvas
    panorama_size = (xmax - xmin, ymax - ymin)
    panorama = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=image1.dtype)

    # Fill the panorama with image1
    panorama[-ymin : h1 - ymin, -xmin : w1 - xmin] = image1  # TODO why minus

    if False:
        for x in range(h1 - ymin, panorama.shape[1]):
            for y in range(w1 - xmin, panorama.shape[0]):
                point = np.array([x, y, 1], dtype=np.float32)
                point2 = homography @ point
                point2 /= point2[2]
                point3 = point2[:2]

                neighboring_point_left_top = np.array(
                    [np.floor(point2[0]), np.floor(point2[1])], dtype=np.int32
                )

                neighboring_point_right_top = np.array(
                    [np.ceil(point2[0]), np.floor(point2[1])], dtype=np.int32
                )
                neighboring_point_left_bottom = np.array(
                    [np.floor(point2[0]), np.ceil(point2[1])], dtype=np.int32
                )
                neighboring_point_right_bottom = np.array(
                    [np.ceil(point2[0]), np.ceil(point2[1])], dtype=np.int32
                )
                calculate_distance_left_top = np.linalg.norm(
                    point3 - neighboring_point_left_top
                )
                calculate_distance_right_top = np.linalg.norm(
                    point3 - neighboring_point_right_top
                )
                calculate_distance_left_bottom = np.linalg.norm(
                    point3 - neighboring_point_left_bottom
                )
                calculate_distance_right_bottom = np.linalg.norm(
                    point3 - neighboring_point_right_bottom
                )

                try:
                    panorama[y, x] = (
                        image2[
                            neighboring_point_left_bottom[1],
                            neighboring_point_left_bottom[0],
                        ]
                        * calculate_distance_left_top
                        + image2[
                            neighboring_point_right_bottom[1],
                            neighboring_point_right_bottom[0],
                        ]
                        * calculate_distance_right_top
                        + image2[
                            neighboring_point_left_top[1], neighboring_point_left_top[0]
                        ]
                        * calculate_distance_left_bottom
                        + image2[
                            neighboring_point_right_top[1],
                            neighboring_point_right_top[0],
                        ]
                        * calculate_distance_right_bottom
                    ) / (
                        calculate_distance_left_top
                        + calculate_distance_right_top
                        + calculate_distance_left_bottom
                        + calculate_distance_right_bottom
                    )
                except IndexError:
                    # If the point is out of bounds, skip it
                    continue

    elif True:
        # Warp image2 to the panorama canvas using the inverse homography
        warped_img2 = cv2.warpPerspective(
            image2, homography, panorama_size, flags=cv2.INTER_LINEAR
        )

        panorama2 = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=image1.dtype)
        # Fill the panorama with image2
        panorama2[-ymin : h2 - ymin, -xmin : w2 - xmin] = warped_img2

        
        # Fill the panorama with the warped image2
        panorama = cv2.add(image1, warped_img2)

    if visualise:
        plt.imshow(panorama)
        plt.title("Warped Image")
        plt.show()

    return panorama


def warp_images(homography, image1, image2, visualise=False):
    # use inverse transform

    # we can compute the inverse homography
    # and use it to avoid the hole fillig problem

    # create a new image with the size of the first image + second image
    # and fill it with undefined values

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    corners_img2 = np.array(
        [[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32
    ).reshape(-1, 1, 2)
    corners_img1 = np.array(
        [[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32
    ).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners_img2, homography)

    all_corners = np.concatenate((warped_corners, corners_img1), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation matrix to shift the image to a positive canvas
    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

    # Warp img2 to panorama
    panorama_size = (xmax - xmin, ymax - ymin)
    warped_img2 = cv2.warpPerspective(image2, translation @ homography, panorama_size)

    # Create empty panorama and paste img1 into it
    panorama = warped_img2.copy()
    panorama[-ymin : h1 - ymin, -xmin : w1 - xmin] = image1

    if visualise:
        plt.imshow(panorama)
        plt.title("Warped Image")
        plt.show()

    return panorama
