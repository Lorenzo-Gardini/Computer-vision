from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._base import _set_random_states


class SequentialAdaBoostClassifier(AdaBoostClassifier):

    def __init__(
        self,
        keras_estimators: list,
        *,
        learning_rate=1.0,
        algorithm="SAMME.R",
        random_state=None,
    ):
        super().__init__(
            estimator=None,
            n_estimators=len(keras_estimators),
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
        )
        self.keras_estimators = keras_estimators
        self.iterator_count = 0

    def _make_estimator(self, append=True, random_state=None):
        estimator = self.keras_estimators[self.iterator_count]
        self.iterator_count += 1
        if random_state is not None:
            _set_random_states(estimator, random_state)
        if append:
            self.estimators_.append(estimator)
        return estimator