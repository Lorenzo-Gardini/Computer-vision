import importlib.util
import cv2
import os
import zipfile
import numpy as np
import pickle

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


def resize_image(image, side):
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


def _compute_rectangle(bounding_box):
    x1, y1, width, height = bounding_box
    x2 = x1 + width
    y2 = y1 + height
    return x1, y1, x2, y2


def highlight_face(image, bounding_box, color):
    x1, y1, x2, y2 = _compute_rectangle(bounding_box)
    return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color, 2)


def crop_face(image, bounding_box):
    x1, y1, x2, y2 = _compute_rectangle(bounding_box)
    return image[y1:y2, x1:x2]


def unzip(path):
    with zipfile.ZipFile(path, "r") as file:
        file.extractall()


def load_binary(path):
    with open(path, "rb") as file:
        result = pickle.load(file)
    return result


def dump_binary(binary, path):
    with open(path, "wb") as file:
        pickle.dump(binary, file)
