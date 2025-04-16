import typing
from typing import List, Tuple
import numpy as np
import cv2


def create_panorama(images: List[np.ndarray]) -> np.ndarray:
    # Create a panorama from a list of images
    # This function will be called in the main function
    # It will take a list of images and return a panorama image

    if len(images) < 2:
        return images[0]
    
    if len(images) == 2:
        # detect feature points

        # match feature points
        # calculate homography via ransac
        # wrap image
        # composite image
        
        pass


    for image in images:
        # set first image as reference
        correspondence_score = 0
        correspondence_image = 1

        # try all n over k possibilites of iamge for the start

        if len(images) > 1:
            for i in range(1, len(images)):
                # detect feature point
                # match feature points
                # calculate homography via ransac
                # calculate correspondence score
                this_score = 0.1  # dummy value
                if this_score > correspondence_score:
                    correspondence_score = this_score

                    correspondence_image = i
                    # set this image as correspondence image

        wrapped_image = None  # dummy value

        composite_image = None  # dummy value

        # remove reference image from list
        images.remove(image)

        # remove correspondence image from list
        images.remove(correspondence_image)

        # store composited_iamge as starting image for nex composition
    return images[0]


def main():
    images = []
    # collect images
    # set refernece image
    # iterate over images

    create_panorama(images)


if __name__ == "__main__":
    main()
