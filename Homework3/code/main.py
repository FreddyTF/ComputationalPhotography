def main():
    images = []
    # collect images
    # set refernece image
    # iterate over images

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
    pass


if __name__ == "__main__":
    main()
