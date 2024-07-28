import sys
import cv2
import numpy as np

def parse_args(argv):
    """ Parse application arguments

    Script execution example:

        python q3.py lena.png 4

    First positional argument is "location". It indicates the path to the image to read.
    Default: "image.png"

    Second position argument is "m". It indicates the number of gray levels.
    Default: 5
    """
    argc = len(argv)
    if argc < 3:
        m = 5
    else:
        m = argv[2]

    if argc < 2:
        location = "image.png"
    else:
        location = argv[1]

    return (location, int(m))

def load_image(location):
    """ Loads the image from the given location """
    image = cv2.imread(location)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def create_palette(min_val, max_val, m):
    """ Create a palette of m equally spread gray values between min_value and max_value """"
    step = (max_val - min_val) // (m - 1)
    palette = []

    while min_val <= max_val:
        palette.append(min_val)
        min_val += step
    
    return palette

def find_closest_palette_color(value, palette):
    """ Find the closest to value palette value """
    left = 0
    right = len(palette) - 1

    while left < right:
        d_left = abs(palette[left] - value)
        d_right = abs(palette[right] - value)

        if d_left <= d_right:
            right -= 1
        else:
            left += 1
    
    return palette[left]

def error_diffusion(image, m):
    """ Perform error diffusion on image with m gray levels """
    rows, cols = image.shape
    image = image.copy().astype(np.int16)
    palette = create_palette(0, 255, m)

    for y in range(rows):
        for x in range(cols):
            old_pixel = image[y, x]
            new_pixel = find_closest_palette_color(old_pixel, palette)

            image[y, x] = new_pixel
            error = old_pixel - new_pixel
            
            if x < cols - 1:
                image[y, x + 1] += error * 7.0 / 16.0

            if y < rows - 1:
                if x > 0:
                    image[y + 1, x - 1] += error * 3.0 / 16.0

                image[y + 1, x] += error * 5.0 / 16.0
                if x < cols - 1:
                    image[y + 1, x + 1] += error * 1.0 / 16.0

    return image.astype(np.uint8)

def main():
    location, m = parse_args(sys.argv)
    image = load_image(location)
    diffused_image = error_diffusion(image, m)
    cv2.imwrite(location.replace("png", "diffused.png"), diffused_image)

main()