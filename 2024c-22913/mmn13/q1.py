"""
The algorithm below performs an adaptive histogram equalization with contrast limiting.

1. It splits the image to strips of equal size
2. Strip by strip, perform histogram equalization with contrast limiting
"""
import cv2
import numpy as np

def limit_contrast(cdf: np.ndarray, contrast_limit: float):
    """ Applies a contrast limit to the given cdf

    The contrast limit is given as a ratio between min and max value.

    Clip the cdf within (0, contrast_limit) and then distribute the excess uniformly on
    the other bins.

    Note, that after filling the excess, it is possible that the cdf will again have bins that are
    above the contrast limit.

    The operation is done in place.
    """
    # convert from ratio to actual value
    contrast_limit = cdf.min() + (cdf.max() - cdf.min()) * contrast_limit

    # clip and track the excess
    excess = 0
    for i, v in enumerate(cdf):
        if v > contrast_limit:
            excess += v - contrast_limit
            cdf[i] = contrast_limit

    # distribute the excess
    cdf[:] = cdf + (excess / cdf.size)

def equalize(arr: np.ndarray, contrast_limit: int):
    """ Equalizes the histogram of the given array arr and apply contrast limiting.

    This operation is done inplace and we use slices to work on difference strips.
    """
    hist, _ = np.histogram(arr, 256, (0, 256))
    cdf = hist.cumsum()
    cdf = cdf / cdf.max() * 255
    limit_contrast(cdf, contrast_limit)
    cdf = cdf / cdf.max() * 255
    arr[:] = cdf[arr]

image = cv2.imread('./assets/embedded-squares.png', cv2.IMREAD_GRAYSCALE)

# limit the contrast to the 80% value (0.8 * 256)
contrast_limit = 0.8

# the size of the strips
kernel_width = 16
kernel_height = 16

for kernel_top in range(0, image.shape[0], kernel_height):
    for kernel_left in range(0, image.shape[1], kernel_width):
        strip = image[kernel_top:(kernel_top+kernel_height), kernel_left:(kernel_left+kernel_width)]
        equalize(strip, contrast_limit)

cv2.imwrite("output.png", image)