from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._base import _set_random_states


class SequentialAdaBoostClassifier(AdaBoostClassifier):

    def __init__(
        self,
        unfit_estimators: list,
        *,
        learning_rate=1.0,
        algorithm="SAMME.R",
        random_state=None,
    ):
        super().__init__(
            estimator=None,
            n_estimators=len(unfit_estimators),
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
        )
        self._unfit_estimators = unfit_estimators
        self._iterator_count = 0

    def _make_estimator(self, append=True, random_state=None):
        estimator = self._unfit_estimators[self._iterator_count]
        self._iterator_count += 1
        if random_state is not None:
            _set_random_states(estimator, random_state)
        if append:
            self.estimators_.append(estimator)
        return estimator
