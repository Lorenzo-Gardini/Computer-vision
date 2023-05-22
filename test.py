from utils import *
import cv2 as cv
from mtcnn import MTCNN
import numpy as np
import json
from classifier import Classifier

# detector = MTCNN()
#
# for day in ["09", "11", "13", "14", "15"]:
#     for i in list(range(1, 9)) + ["parente"]:
#         path = f"puntata_{day}_04_23_{str(i)}.png"
#         image = load_image("resources/soliti_ignoti/" + path)
#         faces = detector.detect_faces(image)
#         sizes = [face["box"][2] * face["box"][3] for face in faces]
#         face_index = np.argmax(sizes)
#         x, y, width, height = faces[face_index]["box"]
#         new_x = x + width // 2 - height // 2
#         left = 0
#         right = 0
#         if new_x < 0:
#             left = -new_x
#             new_x = 0
#         if new_x + height > image.shape[1]:
#             right = new_x + height - image.shape[1]
#         border_image = cv.copyMakeBorder(image, 0, 0, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
#         bb = (new_x, y, height, height)
#         show_image(highlight_face(border_image, bb, (255, 0, 0)))
#         cropped = crop_face(border_image, bb)
#         resized = resize_image(cropped, 224)
#         show_image(resized)
#         cv.imwrite("resources/out/" + path, resized)

episodes = [[], [], [], [], []]
for i, day in enumerate(["09", "11", "13", "14", "15"]):
    for index in list(range(1, 9)) + ["parente"]:
        episodes[i].append(load_image(f"soliti_ignoti/puntata_{day}_04_23_{str(index)}.png"))


def feature_extractor(image):
    return np.zeros((10,))


classifier = Classifier(feature_extractor=feature_extractor)
with open("labels/class_encoding.json", "r") as f:
    class_encoding = json.load(f)
items = list(class_encoding.items())
items.sort(key=lambda x: x[1])
classes = [key for key, _ in items]
for episode in episodes:
    relative_index, relation_index = classifier.play(episode[:8], episode[8])
    print(f"The classifier chose stranger #{relative_index + 1} which was in relation {classes[relation_index]}")
