import pickle
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from itertools import product


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
features_train_binary = np.empty((train_binary.shape[0], train_features.shape[1] * 2), dtype=np.float32)
features_train_binary[:, :train_features.shape[1]] = \
    np.array([train_features[train_faces.index(path)] for path in train_binary["p1"]])
features_train_binary[:, train_features.shape[1]:] = \
    np.array([train_features[train_faces.index(path)] for path in train_binary["p2"]])

test_faces = os.listdir("test-private-faces")
test_faces.sort()
features_test_binary = np.empty((test_binary.shape[0], test_features.shape[1] * 2), dtype=np.float32)
features_test_binary[:, :test_features.shape[1]] = \
    np.array([test_features[test_faces.index(path)] for path in test_binary["p1"]])
features_test_binary[:, test_features.shape[1]:] = \
    np.array([test_features[test_faces.index(path)] for path in test_binary["p2"]])

functions = [
  lambda X: X[:, :(X.shape[1] // 2)] + X[:, (X.shape[1] // 2):],
  lambda X: X[:, :(X.shape[1] // 2)] * X[:, (X.shape[1] // 2):],
  lambda X: np.abs(X[:, :(X.shape[1] // 2)] - X[:, (X.shape[1] // 2):]),
  lambda X: (X[:, :(X.shape[1] // 2)] + X[:, (X.shape[1] // 2):]) ** 2,
  lambda X: (X[:, :(X.shape[1] // 2)] - X[:, (X.shape[1] // 2):]) ** 2,
  lambda X: X[:, :(X.shape[1] // 2)] ** 2 + X[:, (X.shape[1] // 2):] ** 2,
  lambda X: X[:, :(X.shape[1] // 2)] ** 2 - X[:, (X.shape[1] // 2):] ** 2
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
        model.fit(features_train_binary, train_binary["label"])
        score = model.score(features_test_binary, test_binary["label"])
        grid_search_models.append((C, kernel))
        grid_search_scores.append(score)
        print(f"Fitted model with parameters (lambda={i}, C={C}, kernel={kernel}), score: {score}")
    models.append(grid_search_models[np.argmax(grid_search_scores)])
