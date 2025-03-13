from time import time
import typing
from filter import gaussian_1d, gaussian_2d
import numpy as np
from filter import apply_non_separable_filter, apply_separable_filter
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def measure_gaussian(
    image: np.ndarray, size: int, iterations: int = 100, which: str = "1d"
) -> typing.List[float]:
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
            _ = apply_separable_filter(image, kernel)

        elif which == "2d":
            _ = apply_non_separable_filter(image, kernel)

        end_time = time()
        time_elapsed = end_time - start_time
        measurements.append(time_elapsed)
    return measurements


def plot_gaussian():
    image_size = range(10, 500, 10)  # range(10, 1000, 500)

    measurements_1d = {}
    measurements_2d = {}
    for size in image_size:
        random_image = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        measurements_1d[size] = measure_gaussian(
            random_image, size=3, which="1d", iterations=3
        )

        measurements_2d[size] = measure_gaussian(
            random_image, size=3, which="2d", iterations=3
        )

    # Convert measurements to lists for plotting
    sizes = list(measurements_1d.keys())
    avg_runtimes_1d = [np.mean(measurements_1d[size]) for size in sizes]
    avg_runtimes_2d = [np.mean(measurements_2d[size]) for size in sizes]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, avg_runtimes_1d, color="blue", label="1D Gaussian")
    plt.scatter(sizes, avg_runtimes_2d, color="red", label="2D Gaussian")
    plt.title("Average Runtime of Gaussian Filters by Image Size")
    plt.xlabel("Image Size")
    plt.ylabel("Average Runtime (seconds)")
    plt.legend()
    plt.show()


def plot_violin(measurements_1d, measurements_2d):
    # Prepare data for plotting
    sizes = []
    runtimes = []
    types = []

    for size, times in measurements_1d.items():
        sizes.extend([size] * len(times))
        runtimes.extend(times)
        types.extend(["1D"] * len(times))

    for size, times in measurements_2d.items():
        sizes.extend([size] * len(times))
        runtimes.extend(times)
        types.extend(["2D"] * len(times))

    # Create a DataFrame for seaborn
    data = pd.DataFrame({"Size": sizes, "Runtime": runtimes, "Type": types})

    # Plot the violin chart
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Size", y="Runtime", hue="Type", data=data, split=True)
    plt.title("Runtime of Gaussian Filters by Image Size")
    plt.xlabel("Image Size")
    plt.ylabel("Runtime (seconds)")
    plt.legend(title="Filter Type")
    plt.show()


plot_gaussian()
