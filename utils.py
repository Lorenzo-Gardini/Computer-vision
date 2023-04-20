import importlib.util
import cv2
import os

import numpy as np

if importlib.util.find_spec("google") is not None and \
        importlib.util.find_spec("google.colab") is not None and \
        importlib.util.find_spec("google.colab.patches") is not None:
    from google.colab.patches import cv2_imshow


    def show_image(image):
        cv2_imshow(image)
else:
    def show_image(image):
        cv2.imshow("", image)
        cv2.waitKey(0)
        cv2.destroyWindow("")


def load_image(path, mode=cv2.IMREAD_COLOR, dtype=np.uint8):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    else:
        return cv2.imread(path, mode).astype(dtype)


def _resize_image(image, side):
    if not isinstance(side, int):
        raise TypeError()
    old_height, old_width, _ = image.shape
    shape = \
        (side, int(old_height * (side / old_width))) \
            if old_width >= old_height \
            else (int(old_width * (side / old_height)), side)
    return cv2.resize(
        image,
        shape,
        interpolation=(cv2.INTER_AREA if old_height > side or old_width > side else cv2.INTER_CUBIC)
    )


def _compute_borders(shape, side):
    height, width, _ = shape
    top = (side - height) // 2
    bottom = side - height - top
    left = (side - width) // 2
    right = side - width - left
    return top, bottom, left, right


def resize_with_borders(image, side):
    resized_image = _resize_image(image, side)
    top, bottom, left, right = _compute_borders(resized_image.shape, side)
    return cv2.copyMakeBorder(resized_image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)


def pair_images(images):
    return [(image_a, image_b)
            for i, image_a in enumerate(images)
            for j, image_b in enumerate(images) if i < j]


def optionally_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)