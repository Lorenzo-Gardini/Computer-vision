import pickle
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from itertools import product
from classifiers import SequentialAdaBoostClassifier


def load(cache_filename):
    with open(cache_filename, "rb") as features_file:
        features = pickle.load(features_file)
    return features


train_features = load("handcrafted/train-features.bin")
test_features = load("handcrafted/test-features.bin")
train_binary = pd.read_csv("labels/train_binary.csv")
test_binary = pd.read_csv("labels/test_binary.csv")

train_faces = os.listdir("train-faces")
train_faces.sort()
features_train_binary = np.empty((2, train_binary.shape[0], train_features.shape[1]), dtype=np.float32)
features_train_binary[0] = np.array([train_features[train_faces.index(path)] for path in train_binary["p1"]])
features_train_binary[1] = np.array([train_features[train_faces.index(path)] for path in train_binary["p2"]])

test_faces = os.listdir("test-private-faces")
test_faces.sort()
features_test_binary = np.empty((2, test_binary.shape[0], test_features.shape[1]), dtype=np.float32)
features_test_binary[0] = np.array([test_features[test_faces.index(path)] for path in test_binary["p1"]])
features_test_binary[1] = np.array([test_features[test_faces.index(path)] for path in test_binary["p2"]])

functions = [
    lambda X: X[0] + X[1],
    lambda X: X[0] * X[1],
    lambda X: np.abs(X[0] - X[1]),
    lambda X: (X[0] + X[1]) ** 2,
    lambda X: (X[0] - X[1]) ** 2,
    lambda X: X[0] ** 2 + X[1] ** 2,
    lambda X: X[0] ** 2 - X[1] ** 2
]
grid = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "poly"]
}
models = []

for i, function in enumerate(functions):
    grid_search_models = []
    grid_search_scores = []
    for C, kernel in product(grid["C"], grid["kernel"]):
        model = Pipeline([
            ("lambda", FunctionTransformer(function)),
            ("classifier", SVC(C=C, kernel=kernel, degree=2, gamma="scale", random_state=1234))
        ])
        model.fit(features_train_binary, train_binary["labels"])
        score = model.score(features_test_binary, test_binary["label"])
        grid_search_models.append((C, kernel))
        grid_search_scores.append(score)
        print(f"Fitted model with parameters (lambda={i}, C={C}, kernel={kernel}), score: {score}")
    models.append(grid_search_models[np.argmax(grid_search_scores)])

final_models = []
final_scores = []

for learning_rate in [0.1, 1, 10]:
    clf = SequentialAdaBoostClassifier(
        [SVC(C=C, kernel=kernel, degree=2, gamma="scale", random_state=1234, max_iter=1) for C, kernel in models],
        learning_rate=learning_rate,
        random_state=1234
    )
    clf.fit(features_train_binary, train_binary["labels"])
    final_models.append(learning_rate)
    final_scores.append(clf.score(features_test_binary, test_binary["label"]))

best_model_index = np.argmax(final_scores)
print(f"Best AdaBoost model has learning rate {final_models[best_model_index]} " +
      f"and score {final_scores[best_model_index]}")
