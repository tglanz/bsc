import cv2 as cv
import numpy as np


FONT = cv.FONT_HERSHEY_SIMPLEX
IMAGE_SIZE=256
IMAGE_MID=IMAGE_SIZE // 2
IMAGE_CENTER=(IMAGE_MID, IMAGE_MID)
RGB_WHITE=(255, 255, 255)
RGB_BLACK=(0, 0, 0)
RGB_REVELJS_BG=(25,25,25)
RGB_BG=RGB_REVELJS_BG

BORDER_THIN=1
BORDER_MEDIUM=2
BORDER_THICK=3

FONT_SMALL=2

RGB_PRIMARY=RGB_WHITE
RGB_SECONDARY=(255, 0, 0)

def new_image():
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    image[:] = RGB_BG
    return image

def save_image(name, image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cv.imwrite(f"dist/{name}.png", image)

def render_prologue_pre_cover():
    image = new_image()

    cv.circle(image, IMAGE_CENTER, IMAGE_MID,  RGB_PRIMARY, BORDER_MEDIUM)

    save_image("prologue-pre-cover", image)

    # cv.line(image, (0, 0), (24, 24), (0, 0, 0), 1)
    # cv.circle(image, (12, 12), 12, (0, 0, 0), 1)
    # cv.rectangle(image, (4, 4), (16, 16), (0, 0, 0), 1)

    # cv.putText(image, 'Hi', (13, 13), font, 4, (255, 0, 0), 2, cv.LINE_AA)

def render_prologue_post_cover():
    image = new_image()

    cv.circle(image, IMAGE_CENTER, IMAGE_MID,  RGB_PRIMARY, BORDER_MEDIUM)

    count = 8 
    height = IMAGE_SIZE // count 
    for i in range(10):

        top = i * height
        bot = ((i + 1) * height) - 1
        left = 0
        right = IMAGE_SIZE - 1

        cv.rectangle(image, (left, top), (right, bot), RGB_PRIMARY, BORDER_THIN)

    save_image("prologue-post-cover", image)

def main():
    render_prologue_pre_cover()
    render_prologue_post_cover()
   
if __name__ == '__main__':
    main()
