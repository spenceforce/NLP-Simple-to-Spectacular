"""OneR classifier.

Predict the most common label for each category in the most informative feature.
"""

import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import unique_labels


class Rule(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        """Train on the categories in the first column of `X`."""
        # Convert to `numpy` arrays for consistency.
        X, y = np.array(X), np.array(y)

        # Store the classes.
        # sklearn provides a handy function `unique_labels` for this
        # purpose. You could also use `np.unique`.
        self.classes_ = unique_labels(y)

        predictors = {}
        # Get the unique categories from the first column.
        categories = np.unique(X[:, 0])
        for value in categories:
            # Create a boolean array where `True` indices indicate the
            # rows where value is `value`.
            is_value = X[:, 0] == value

            # Grab all data points and labels in this value.
            _X = X[is_value]
            _y = y[is_value]

            # Train a baseline classifier on the value.
            predictors[value] = DummyClassifier().fit(_X, _y)

        self.predictors_ = predictors

        # Create a fallback predictor for unknown categories.
        self.unknown_predictor_ = DummyClassifier().fit(X, y)
        return self

    def predict(self, X):
        """Predict the labels for inputs `X`."""
        X = np.array(X)

        # Create an empty array that will hold the predictions.
        rv = np.zeros(X.shape[0])

        # Get the unique categories from the first column.
        categories = np.unique(X[:, 0])
        for value in categories:
            # Create a boolean array where `True` indices indicate the
            # rows where value is `value`.
            is_value = X[:, 0] == value

            # Grab all data points in this value.
            _X = X[is_value]

            # Predict the label for all datapoints in `_X`.
            try:
                predictions = self.predictors_[value].predict(_X)
            except KeyError:
                # Fallback to the predictor for unknown categories.
                predictions = self.unknown_predictor_.predict(_X)

            # Assign the prediction for this value to
            # the corresponding indices in `rv`.
            rv[is_value] = predictions
        return rv


class OneR(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        """Find the best rule in the dataset."""
        # Sanity check on `X` and `y`.
        X, y = validate_data(self, X, y, accept_sparse=True)

        col_idx = score = rule = None
        # Iterate over the indices for each column in X.
        for i in range(X.shape[1]):
            # Create a new matrix containing just the ith column.
            _X = X[:, [i]]
            # Convert sparse matrix to `numpy` array.
            # `Rule` works on numpy arrays, so we should use consistent
            # array types.
            if scipy.sparse.issparse(_X):
                _X = _X.toarray()

            # Create a rule for the ith column.
            rule_i = Rule().fit(_X, y)
            # Score the ith columns accuracy.
            score_i = rule_i.score(_X, y)

            # Keep the rule for the ith column if it has the highest
            # accuracy so far.
            if score is None or score_i > score:
                rule = rule_i
                score = score_i
                col_idx = i

        self.rule_ = rule
        self.i_ = col_idx
        return self

    def predict(self, X):
        """Predict the labels for inputs `X`."""
        # Sanity check on `X`.
        X = validate_data(self, X, reset=False, accept_sparse=True)
        _X = X[:, [self.i_]]
        # Convert sparse matrix to `numpy` array.
        # `Rule` works on numpy arrays, so we should use consistent
        # array types.
        if scipy.sparse.issparse(_X):
            _X = _X.toarray()

        # Return predictions for the rule.
        return self.rule_.predict(_X)
