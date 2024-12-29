"""Bag of characters feature extractor.

Convert string to character counts.
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class BagOfChars(TransformerMixin, BaseEstimator):
    """Bag of characters feature extractor."""

    def fit(self, X, y=None):
        """Fit on all characters in the array `X`.

        Note: `X` should be a 1d array.
        """
        # We run our own validation check since `validate_data`
        # expects a 2d numberic array.
        # We want a 1d text array so we'll check its shape here.
        # While iterating over the array values we'll check
        # they are text while trying to extract characters.
        assert len(X.shape) == 1

        vocabulary_ = {}
        # Iterate over each string in the array.
        for x in X:
            # Check it's a string!
            assert isinstance(x, str)
            # Get the unique characters in the string.
            chars = np.unique(list(x))
            # Add each character to the vocabulary if it isn't
            # there already.
            for char in chars:
                if char not in vocabulary_:
                    vocabulary_[char] = len(vocabulary_)
        self.vocabulary_ = vocabulary_
        return self

    def transform(self, X):
        """Transform `X` to a count matrix.

        Note: `X` should be a 1d array.
        """
        # Run our own checks.
        assert len(X.shape) == 1
        # Check we fit the instance.
        assert hasattr(self, "vocabulary_")

        # Create a matrix to hold the counts.
        rv = np.zeros((X.shape[0], len(self.vocabulary_)))
        # Iterate over each string in the array.
        for i, x in enumerate(X):
            # Check it's a string!
            assert isinstance(x, str)
            # Get the unique characters in the string and their
            # counts.
            chars, counts = np.unique(list(x), return_counts=True)
            # Add each character count to the count matrix
            # for the specific row.
            for char, count in zip(chars, counts):
                # Make sure the character is part of the vocabulary,
                # otherwise ignore it.
                if char in self.vocabulary_:
                    rv[i, self.vocabulary_[char]] = count
        # Return the count matrix.
        return rv
