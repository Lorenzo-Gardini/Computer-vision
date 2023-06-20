import pickle
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier
def load(cache_filename):
    with open(cache_filename, "rb") as features_file:
        features = pickle.load(features_file)
    return features

train_features = load("handcrafted/train-features-full.bin")
test_features = load("handcrafted/test-features-full.bin")
train_binary = pd.read_csv("labels/train_binary.csv")
test_binary = pd.read_csv("labels/test_binary.csv")
test_faces = os.listdir("resources/test-private-faces")
train_faces = os.listdir("resources/train-faces")

#features_train_binary = load("downsampled/downsampled-train-features-2056.bin")

#train_binary = train_binary[::5]
train_binary = pd.concat([train_binary[:3], train_binary[-3:]])
test_binary = pd.concat([test_binary[:3], test_binary[-3:]])

train_faces.sort()
features_train_binary = np.empty((train_binary.shape[0], train_features.shape[1] * 2), dtype=np.float32)
features_train_binary[:, :train_features.shape[1]] = \
     np.array([train_features[train_faces.index(path)] for path in train_binary["p1"]])
features_train_binary[:, train_features.shape[1]:] = \
     np.array([train_features[train_faces.index(path)] for path in train_binary["p2"]])


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


print('start grid')

functions = [sum, prod, abs_diff, squared_sum, squared_diff, sum_squares, diff_squares]

grid_params = {
    'min_child_weight': [1, 5, 10],
    # 'gamma': [0, 1, 3, 5, 7, 9],
    # 'subsample': [0.6, 0.8, 1.0],
    # 'colsample_bytree': [0.6, 0.8, 1.0],
    # 'max_depth': [3, 6, 9, 12, 15],
    # 'n_estimators': [100, 200, 300],
}
base_XGBoost = XGBClassifier(objective='binary:logistic', 
                             eval_metric='error', 
                             #tree_method='gpu_hist',
                             nthread=3, 
                             seed=1234)
best_params = []

grid_search_params = {'classifier__'+grid_params_name: grid_params_value for grid_params_name, grid_params_value in grid_params.items()}
for function in functions[:1]:
    estimator = Pipeline([
        ("lambda", FunctionTransformer(function)),
        ("classifier", base_XGBoost)
    ])    
    model = GridSearchCV(estimator=estimator,
                         param_grid=grid_search_params,
                         cv=3,
                         scoring="accuracy")
    
    model.fit(features_train_binary, train_binary["label"])
    print(f'SCORE: {model.best_score_}, PARAMS: {model.best_params_}')
    best_algo_params = {key.split('__')[1]: value for key, value in model.best_params_.items()}
    best_params.append(best_algo_params)

best_models = [
    Pipeline([
        ("lambda", FunctionTransformer(function)),
        ("classifier", XGBClassifier(**params) )
    ]) for function, params in zip(functions, best_params)
]

final_grid_search_params = {'final_estimator__'+grid_params_name: grid_params_value for grid_params_name, grid_params_value in grid_params.items()}

stack_grid_search = GridSearchCV(
    StackingClassifier(
      estimators=[(str(i), best_model) for i, best_model in enumerate(best_models)],
      final_estimator=base_XGBoost,
      cv=2,
    ),
    final_grid_search_params,
    verbose=2,
    cv=2,
)


stack_grid_search.fit(features_train_binary, train_binary["label"])
stack_results = pd.DataFrame(stack_grid_search.cv_results_).sort_values(by=["rank_test_score"])
print("Stacking results are in")
stack_results.to_csv("results_stacking.csv")
best_stack_model = stack_grid_search.best_estimator_
print("Best stacking score:")
print(best_stack_model.score(features_test_binary, test_binary["label"]))
# grid = {
#     "lambda__func": [sum, prod, abs_diff, squared_sum, squared_diff, sum_squares, diff_squares],
#     "classifier__C": [0.1, 1, 10],
#     "classifier__kernel": ["rbf", "poly"]
# }
# base_model = Pipeline([
#     ("lambda", FunctionTransformer()),
#     ("classifier", SVC(degree=2, gamma="scale", random_state=1234))
# ])
# grid_search = GridSearchCV(base_model, grid, n_jobs=-1, cv=3, refit=False, verbose=4)
# grid_search.fit(features_train_binary, train_binary["label"])
# results = pd.DataFrame(grid_search.cv_results_).sort_values(by=["rank_test_score"])
# print("Classifiers results are in")
# results.to_csv("results_svc.csv")
#
# best_models = [
#     Pipeline([
#         ("lambda", FunctionTransformer(params["lambda__func"])),
#         ("classifier", SVC(C=params["classifier__C"], kernel=params["classifier__kernel"], degree=2, gamma="scale",
#                            random_state=1234))
#     ]) for params in results[:10]["params"]
# ]
# print("Top 10 best models score:")
# for best_model in best_models:
#     best_model.fit(features_train_binary, train_binary["label"])
#     print(best_model.score(features_test_binary, test_binary["label"]))
#
# stack_grid = {
#   "final_estimator__C": [0.1, 1, 10],
#   "final_estimator__kernel": ["rbf", "poly"]
# }
# stack_grid_search = GridSearchCV(
#     StackingClassifier(
#       estimators=[(str(i), best_model) for i, best_model in enumerate(best_models)],
#       final_estimator=SVC(degree=2, gamma="scale", random_state=1234),
#       cv=3,
#       n_jobs=-1,
#       verbose=4
#     ),
#     stack_grid,
#     cv=3,
#     verbose=4,
#     n_jobs=-1
# )
# stack_grid_search.fit(features_train_binary, train_binary["label"])
# stack_results = pd.DataFrame(stack_grid_search.cv_results_).sort_values(by=["rank_test_score"])
# print("Stacking results are in")
# stack_results.to_csv("results_stacking.csv")
# best_stack_model = stack_grid_search.best_estimator_
# print("Best stacking score:")
# print(best_stack_model.score(features_test_binary, test_binary["label"]))
