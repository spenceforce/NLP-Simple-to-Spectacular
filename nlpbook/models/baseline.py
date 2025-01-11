"""Baseline classifier.

Predict the most common label in the dataset.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BaselineClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        """Train the model with inputs `X` on labels `y`."""
        # Get the unique labels and their counts.
        self.classes_, counts = np.unique(y, return_counts=True)
        # Keep the most common label for prediction.
        # Note we changed the `prediction` attribute to include a
        # trailing suffix because it results from a computation
        # that persists across method calls.
        self.prediction_ = labels[np.argmax(counts)]
        return self

    def predict(self, X):
        """Predict the labels for inputs `X`."""
        # Return the most common label as the prediction for every
        # input.
        return np.full(len(X), self.prediction_, dtype=int)
