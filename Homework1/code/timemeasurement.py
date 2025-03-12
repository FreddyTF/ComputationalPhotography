from time import time
import typing
from filter import gaussian_1d, gaussian_2d


def measure_separable_1d_gaussian(iterations: int = 100, which: str = "1d", size: int) -> typing.List[float]:
    """
    Measure the time it takes to create a 1D Gaussian kernel.
    :param iterations: The number of iterations to run
    :param which: The type of Gaussian kernel to create
    :return: A list of time measurements
    """

    if which == "1d":
        kernel = gaussian_1d(sigma=1, size=size)
    elif which == "2d":
        kernel = gaussian_2d(sigma=1, size=size)
    else:
        raise ValueError("Invalid value for 'which'")
    

    measurements = []
    for i in range(iterations):

        start_time = time()
        if which == "1d":
            kernel = gaussian_1d(sigma=1, size=size)
        elif which == "2d":
            kernel = gaussian_2d(sigma=1, size=size)

        end_time = time()
        time_elapsed = end_time - start_time
        measurements.append(time_elapsed)
    return measurements



