import keras
from keras.layers import Dense, Lambda
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from utils import *
from detectors import *
from classifiers import SequentialAdaBoostClassifier
import numpy as np

landscape = load_image("resources/family.jpeg")
landscape_resized = resize_with_borders(landscape, 512)
print(landscape_resized.shape)
show_image(landscape_resized)

portrait = load_image("resources/portrait.jpg")
portrait_resized = resize_with_borders(portrait, 512)
print(portrait_resized.shape)
show_image(portrait_resized)

detector = FaceDetector(landscape)
landscape_highlighted = detector.highlight_faces()
show_image(landscape_highlighted)
for face in detector.crop_faces():
    show_image(face)
for pair in pair_images(detector.crop_faces()):
    show_image(np.hstack([resize_with_borders(pair[0], 100), resize_with_borders(pair[1], 100)]))

model1 = Sequential([
    keras.Input(1),
    Lambda(lambda x: x * 2),
    Dense(50),
    Dense(1)
])
model2 = Sequential([
    keras.Input(1),
    Lambda(lambda x: x / 2),
    Dense(50),
    Dense(1)
])
model1.compile(loss="mse")
model2.compile(loss="mse")
clf = SequentialAdaBoostClassifier(
    [
        KerasClassifier(model=model1, epochs=100, random_state=0xDEADBEEF),
        KerasClassifier(model=model2, epochs=100, random_state=0xDEADBEEF)
    ],
    learning_rate=0.01,
    random_state=0xDEADBEEF
)
clf.fit([[0], [100], [0], [100]], [0, 1, 0, 1])
print(clf.estimators_[0].model.summary())
print("\n")
print(clf.estimators_[1].model.summary())
print("\n")
print(clf.predict([[0], [100]]))
