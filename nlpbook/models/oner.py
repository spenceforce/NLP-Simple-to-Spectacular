"""OneR classifier.

Predict the most common label for each category in the most informative feature.
"""

import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.multiclass import unique_labels


class Rule(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        """Train on the categories in the first column of `X`."""
        # Store the classes.
        # sklearn provides a handy function `unique_labels` for this
        # purpose. You could also use `np.unique`.
        self.classes_ = unique_labels(y)

        predictors = {}
        # Get the unique categories from `X`.
        categories = np.unique(X)
        for value in categories:
            # Create a boolean array where `True` indices indicate the
            # rows that have this value.
            is_value = X == value

            # Grab all data points and labels with this value.
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
        # Create an empty array that will hold the predictions.
        rv = np.zeros(len(X), dtype=int)

        # Get the unique categories from `X`.
        categories = np.unique(X)
        for value in categories:
            # Create a boolean array where `True` indices indicate the
            # rows that have this value.
            is_value = X == value

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
        self.classes_ = unique_labels(y)

        col_idx = score = rule = None

        # Iterate over each feature.
        # `numpy` and `scipy` iterate over rows. Rows are data points.
        # We want the columns (features). An easy trick to iterate
        # over columns is to transpose the matrix which flips it along
        # its diagonal.
        for i, column in enumerate(X.T):
            # Convert sparse matrix to `numpy` array.
            # `Rule` works on numpy arrays, so we should use consistent
            # array types.
            if scipy.sparse.issparse(column):
                column = column.toarray()

            # `column` has matrix shape (1, N) but we need array shape (N,).
            # Use `np.squeeze` to flatten to 1D array.
            column = column.squeeze()

            # Create a rule for the ith column.
            rule_i = Rule().fit(column, y)
            # Score the ith columns accuracy.
            score_i = rule_i.score(column, y)

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
        # Get the ith column from the matrix.
        column = X[:, self.i_]
        # Convert sparse matrix to `numpy` array.
        # `Rule` works on numpy arrays, so we should use consistent
        # array types.
        if scipy.sparse.issparse(column):
            column = column.toarray()

        # `column` has matrix shape (1, N) but we need array shape (N,).
        # Use `np.squeeze` to flatten to 1D array.
        column = column.squeeze()

        # Return predictions for the rule.
        return self.rule_.predict(column)
