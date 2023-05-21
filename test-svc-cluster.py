import pickle
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier


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


def sum(X):
    return X[:, :(X.shape[1] // 2)] + X[:, (X.shape[1] // 2):]


def prod(X):
    return X[:, :(X.shape[1] // 2)] * X[:, (X.shape[1] // 2):]


def abs_diff(X):
    return np.abs(X[:, :(X.shape[1] // 2)] - X[:, (X.shape[1] // 2):])


def squared_sum(X):
    return (X[:, :(X.shape[1] // 2)] + X[:, (X.shape[1] // 2):]) ** 2


def squared_diff(X):
    return (X[:, :(X.shape[1] // 2)] - X[:, (X.shape[1] // 2):]) ** 2


def sum_squares(X):
    return X[:, :(X.shape[1] // 2)] ** 2 + X[:, (X.shape[1] // 2):] ** 2


def diff_squares(X):
    return X[:, :(X.shape[1] // 2)] ** 2 - X[:, (X.shape[1] // 2):] ** 2


grid = {
    "lambda__func": [sum, prod, abs_diff, squared_sum, squared_diff, sum_squares, diff_squares],
    "classifier__C": [0.1, 1, 10],
    "classifier__kernel": ["rbf", "poly"]
}
base_model = Pipeline([
    ("lambda", FunctionTransformer()),
    ("classifier", SVC(degree=2, gamma="scale", random_state=1234, max_iter=1))
])
grid_search = GridSearchCV(base_model, grid, n_jobs=-1, cv=3, refit=False, verbose=4)
grid_search.fit(features_train_binary, train_binary["label"])
results = pd.DataFrame(grid_search.cv_results_).sort_values(by=["rank_test_score"])
print("Classifiers results:")
print(results)

best_models = [
    Pipeline([
        ("lambda", FunctionTransformer(params["lambda__func"])),
        ("classifier", SVC(C=params["classifier__C"], kernel=params["classifier__kernel"], degree=2, gamma="scale",
                           random_state=1234, max_iter=1))
    ]) for params in results[:10]["params"]
]
print("Top 10 best models score:")
for best_model in best_models:
    best_model.fit(features_train_binary, train_binary["label"])
    print(best_model.score(features_test_binary, test_binary["label"]))

stack_grid = {
  "final_estimator__C": [0.1, 1, 10],
  "final_estimator__kernel": ["rbf", "poly"]
}
stack_grid_search = GridSearchCV(
    StackingClassifier(
      estimators=[(str(i), best_model) for i, best_model in enumerate(best_models)],
      final_estimator=SVC(degree=2, gamma="scale", random_state=1234, max_iter=1),
      cv=3,
      n_jobs=-1,
      verbose=4
    ),
    stack_grid,
    cv=3,
    verbose=4,
    n_jobs=-1
)
stack_grid_search.fit(features_train_binary, train_binary["label"])
stack_results = pd.DataFrame(stack_grid_search.cv_results_).sort_values(by=["rank_test_score"])
print("Stacking results:")
print(stack_results)
best_stack_model = stack_grid_search.best_estimator_
print("Best stacking score:")
print(best_stack_model.score(features_test_binary, test_binary["label"]))
