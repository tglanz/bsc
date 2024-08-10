# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(image_path):
    bgr = cv2.imread(image_path)
    bgr = cv2.resize(bgr, (512, int(512 * bgr.shape[0] / bgr.shape[1])))
    bgr = cv2.blur(bgr, (4, 4))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = gray / 255
    return gray

def create_laplacian_kernel(shape):
    P, Q = shape
    kernel = np.zeros((P, Q), dtype=np.float32)
    P_half = P / 2
    Q_half = Q / 2
    PI_square = np.pi * np.pi

    for u in range(P):
        for v in range(Q):
            d_uv = (u - P_half)**2 + (v - Q_half)**2
            kernel[u, v] = -4 * PI_square * d_uv

    return kernel

def normalize_tensor(tensor, minval, maxval):
    old_diff = tensor.max() - tensor.min()
    new_diff = maxval - minval
    return (tensor - tensor.min()) / old_diff * new_diff - 1

for image_name in ["moon", "dogs"]:
    # load the image
    image_path = f"../assets/{image_name}.jpg"
    image_spatial = load_image(image_path)

    # fft transform the image and shift it such that (0, 0) is in the center
    image_freq = np.fft.fft2(image_spatial)
    image_freq = np.fft.fftshift(image_freq)

    # create the laplacian kernel in the frequency domain
    kernel_freq = create_laplacian_kernel(image_spatial.shape)

    # calculate the laplacian and normalize it
    laplacian = kernel_freq * image_freq
    laplacian = np.fft.ifft2(np.fft.ifftshift(laplacian))
    laplacian = np.real(laplacian)
    laplacian = normalize_tensor(laplacian, -1, 1)

    # compute the enhanced image
    enhanced = image_spatial + (-1) * laplacian
    enhanced = np.clip(enhanced, 0, 1)
    enhanced *= 255

    # plot and compare the results
    plt.subplots(1, 2)
    plt.tight_layout()
    plt.suptitle(f"../assets/{image_name}.jpg")
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Original")
    plt.imshow(image_spatial, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Enhanced")
    plt.imshow(enhanced, cmap="gray")
    plt.savefig(f"../assets/{image_name}.comparison.png")