"""Categorical encoder.

Assign unique integers to each category.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data


class CategoricalEncoder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        """Generate numeric categories from `X`.

        Note: All `fit` methods must accept a `y` argument whether
              they use them or not. Transfomers typically ignore
              this argument whether it's passed in or not.
        """
        # Validate the data as before. `skip_check_array=True`
        # tells `validate_data` not to convert `X` to a numeric array
        # type. This is important since we have to deal with numeric
        # or text types.
        X = validate_data(self, X, skip_check_array=True)
        try:
            # Since `validate_data` did not convert `X` to a numeric
            # array, we need to convert it to a matrix if it's still
            # a `DataFrame`.
            X = X.to_numpy()
        except AttributeError:
            # This is not a `DataFrame`. Assume it's a `numpy` or
            # `scipy` array.
            pass

        categories = []
        # Iterate over each column in `X`.
        for column in X.T:
            # Get all unique values in the column.
            values = np.unique(column)
            # Store the unique values as the ith element in the array.
            categories.append(values)
        # Save the categories on the transformer.
        self.categories_ = categories
        return self

    def transform(self, X):
        """Return the categorical values."""
        X = validate_data(self, X, skip_check_array=True, reset=False)
        try:
            X = X.to_numpy()
        except AttributeError:
            pass

        # Create an array with the same shape as `X` to store the
        # categorical values. An unknown category, `-1` is used as
        # the default value.
        rv = np.full(X.shape, -1)
        # Iterate over each column in `X`.
        for i, x in enumerate(X.T):
            # Grab the categories for the ith column of `X`.
            categories = self.categories_[i]
            # Reshape the column to be a Nx1 matrix. Using a matrix
            # instead of an array allows us to leverage `numpy`
            # broadcasting.
            x = x.reshape(-1, 1)
            # Create boolean matrix where `True` values indicate the
            # index of `categories` that equals `x` for each row.
            is_category = x == categories
            # Find the indices of `x` that contain known categories.
            # This tells us which rows have known categories.
            known_category = is_category.any(axis=1)
            # Get the index of the `True` value in each row. This
            # is the numeric value for the category.
            category_value = np.where(is_category)[1]
            # Assign the category index to the appropriate rows.
            rv[known_category, i] = category_value
        return rv
