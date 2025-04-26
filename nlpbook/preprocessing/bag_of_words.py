import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


class BagOfWords(TransformerMixin, BaseEstimator):
    """Bag of words feature extractor."""

    def fit(self, X, y=None):
        """Fit on all characters in the array `X`.

        Note: `X` should be a 1d array.
        """
        # We want a 1d text array so we'll check its shape here.
        # While iterating over the array values we'll check
        # they are text while trying to extract words.
        assert len(X.shape) == 1

        vocabulary_ = {}
        # Iterate over each string in the array.
        for x in X:
            # Check it's a string!
            assert isinstance(x, str)

            # Get the unique words in the string.
            chars = np.unique(x.split())

            # Add each word to the vocabulary if it isn't
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

        # Create a matrix to hold the counts.
        # Due to the number of words in the vocabulary we need to use a
        # sparse matrix.
        # Sparse matrices are space efficient representations of matrices
        # that conserve space by not storing 0 values.
        # They are constructed a bit differently from `numpy` arrays.
        # We'll store the counts and their expected row, col indices in
        # lists that `csr_matrix` will use to construct the sparse matrix.
        row_indices = []
        col_indices = []
        values = []
        # Iterate over each string in the array.
        for i, x in enumerate(X):
            # Check it's a string!
            assert isinstance(x, str)

            # Get the unique words in the string and their
            # counts.
            words, counts = np.unique(x.split(), return_counts=True)
            # Update the running list of counts and indices.
            for word, count in zip(words, counts):
                # Make sure the word is part of the vocabulary,
                # otherwise ignore it.
                if word in self.vocabulary_:
                    values.append(count)
                    row_indices.append(i)
                    col_indices.append(self.vocabulary_[word])

        # Return the count matrix.
        return csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(X.shape[0], len(self.vocabulary_)),
        )
