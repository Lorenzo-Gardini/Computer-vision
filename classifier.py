from typing import Callable, Tuple
from sklearn.dummy import DummyClassifier
from sklearn.base import ClassifierMixin
from keras import Sequential
import pandas as pd
import numpy as np


class Classifier:

    def __init__(
            self,
            relatives_classifier: ClassifierMixin | Sequential | None = None,
            relation_classifier: ClassifierMixin | Sequential | None = None,
            feature_extractor: Callable[[np.ndarray], np.ndarray] | None = None
    ) -> None:
        if relatives_classifier is None:
            self._relatives_classifier = DummyClassifier().fit(None, pd.read_csv("labels/train_binary.csv")["label"])
        else:
            self._relatives_classifier = relatives_classifier
        if relation_classifier is None:
            self._relation_classifier = DummyClassifier().fit(None, pd.read_csv("labels/train_multiclass.csv")["relation"])
        else:
            self._relation_classifier = relation_classifier
        self._feature_extractor = feature_extractor

    def play(self, strangers: list[np.ndarray], relative: np.ndarray) -> Tuple[int, int]:
        if self._feature_extractor is not None:
            relative_features = self._feature_extractor(relative)
            features = np.array(
                [np.concatenate([self._feature_extractor(stranger), relative_features]) for stranger in strangers]
            )
            relative_index = np.argmax(self._relatives_classifier.predict_proba(features)[:, 1])
            relation_index = np.argmax(self._relation_classifier.predict_proba(features[relative_index]))
            return int(relative_index), int(relation_index)
        else:
            relative_index = np.argmax(
                np.vstack([self._relatives_classifier.predict([stranger, relative]) for stranger in strangers])[:, 1]
            )
            relation_index = np.argmax(self._relation_classifier.predict([strangers[relative_index], relative]))
            return int(relative_index), int(relation_index)
