import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.multiclass import unique_labels


class Split(ClassifierMixin, BaseEstimator):
    """Split the data on the feature value."""

    def __init__(self, idx, value):
        # Index of the feature matrix.
        self.idx = idx
        # Value to split the data on.
        self.value = value

    def fit(self, X, y):
        # Convert other data types to numpy array
        # for consistency.
        X, y = np.array(X), np.array(y)

        # Grab class labels.
        self.classes_ = unique_labels(y)

        # Create boolean arrays to split the groups on.
        rhs = X[:, self.idx] >= self.value
        lhs = ~rhs

        # Create baseline classifiers for each split.
        self.lhs_ = DummyClassifier().fit(X[lhs], y[lhs])
        self.rhs_ = DummyClassifier().fit(X[rhs], y[rhs])

        return self

    def predict(self, X):
        # Convert other data types to numpy array
        # for consistency.
        X = np.array(X)

        # Make our empty prediction array.
        pred = np.zeros(X.shape[0], dtype=int)

        # Create boolean arrays to split the groups on.
        rhs = X[:, self.idx] >= self.value
        lhs = ~rhs

        # Populate the prediction array with predictions from
        # each group.
        if lhs.sum() > 0:
            pred[lhs] = self.lhs_.predict(X[lhs])
        if rhs.sum() > 0:
            pred[rhs] = self.rhs_.predict(X[rhs])

        return pred


def find_best_split(X, y):
    """Iterate over all possible values in `X` to find the best
    split point."""
    # Convert other data types to numpy array
    # for consistency.
    X, y = np.array(X), np.array(y)

    # Variables for the two groups.
    best_split = best_score = None

    # Iterate over each feature.
    for i, column in enumerate(X.T):
        # Iterate over each unique value in column.
        for value in np.unique(column):
            try:
                split = Split(i, value).fit(X, y)
            except ValueError:
                # `DummyClassifier` will raise a `ValueError`
                # if it is trained on an empty dataset, in which
                # case we just skip this split.
                continue

            # Score the split on this value.
            score = split.score(X, y)

            # Keep this split if it has the best score so far.
            if best_score is None or score > best_score:
                best_split = split
                best_score = score

    # Raise an error if there is no way to split the data.
    if best_split is None:
        raise ValueError

    return best_split


def find_best_splits(X, y):
    """Generate a binary tree based on the data."""
    # Create a baseline classifier for the entire dataset.
    unsplit = DummyClassifier().fit(X, y)
    try:
        # Create a split on the dataset.
        split = find_best_split(X, y)
    except ValueError:
        # If it's impossible to split the dataset, return
        # the baseline classifier.
        return unsplit

    # If the baseline classifier performs better than the
    # split classifier, return the baseline classifier.
    if unsplit.score(X, y) >= split.score(X, y):
        return unsplit

    # Create boolean arrays for each subset of the data based
    # on the split value.
    rhs = X[:, split.idx] >= split.value
    lhs = ~rhs

    # Recursively update the left hand side classifier.
    split.lhs_ = find_best_splits(X[lhs], y[lhs])
    # Recursively update the right hand side classifier.
    split.rhs_ = find_best_splits(X[rhs], y[rhs])

    # Return the updated split.
    return split


class DecisionTree(ClassifierMixin, BaseEstimator):
    """Binary decision tree classifier."""

    def fit(self, X, y):
        # Convert sparse matrix to `numpy` matrix.
        if issparse(X):
            X = X.toarray()
        # Convert `X` and `y` to `numpy` arrays for consistency.
        X, y = np.array(X), np.array(y)

        # Grab the labels.
        self.classes_ = unique_labels(y)

        # Create the binary tree.
        self.tree_ = find_best_splits(X, y)

        return self

    def predict(self, X):
        # Convert sparse matrix to `numpy` matrix.
        if issparse(X):
            X = X.toarray()
        # Convert `X` to `numpy` array for consistency.
        X = np.array(X)
        # Return predictions from the binary decision tree.
        return self.tree_.predict(X)
