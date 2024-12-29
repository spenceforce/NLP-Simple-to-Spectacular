"""OneR classifier.

Predict the most common label for each category in the most informative feature.
"""

import numpy as np
import scipy.sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import validate_data


class Rule(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        """Find the most predictive rule."""
        # Sanity check on `X` and `y`.
        X, y = validate_data(self, X, y)
        predictors = {}
        # Get the unique categories from the first column.
        categories = np.unique(X[:, 0])
        for category in categories:
            # Create a conditional array where each index
            # is a boolean indicating if that index in the
            # first column of `X` is the category we're iterating
            # over.
            is_category = X[:, 0] == category
            # Grab all data points and labels in this category.
            _X = X[is_category]
            _y = y[is_category]
            # Train a baseline classifier on the category.
            predictors[category] = DummyClassifier().fit(_X, _y)
        self.predictors_ = predictors
        # Create a fallback predictor for unknown categories.
        self.unknown_predictor_ = DummyClassifier().fit(X, y)
        return self

    def predict(self, X):
        """Predict the labels for inputs `X`."""
        # Sanity check on `X`.
        # `reset` should be `True` in `fit` and `False` everywhere else.
        X = validate_data(self, X, reset=False)
        # Create an empty array that will hold the predictions.
        rv = np.zeros(X.shape[0])
        # Get the unique categories from the first column.
        categories = np.unique(X[:, 0])
        for category in categories:
            # Create a conditional array where each index
            # is a boolean indicating if that index in the
            # first column of `X` is the category we're iterating
            # over.
            is_category = X[:, 0] == category
            # Grab all data points in this category.
            _X = X[is_category]
            # Predict the label for all datapoints in `_X`.
            try:
                predictions = self.predictors_[category].predict(_X)
            except KeyError:
                # Fallback to the predictor for unknown categories.
                predictions = self.unknown_predictor_.predict(_X)
            # Assign the prediction for this category to
            # the corresponding indices in `rv`.
            rv[is_category] = predictions
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
