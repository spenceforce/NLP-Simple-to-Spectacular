import numpy as np
from scipy.sparse import csr_array, sparray


class nlp_array(sparray):
    """
    Sparse array that supports simple N dimension operations.

    The sparse array implementation is based on `scipy.sparse.dok_array` but uses a recursive
    data structure to handle each dimension.
    """

    def __init__(self, shape, dtype=float):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self._values = {}

    def __getitem__(self, item):
        """N-D sparse array indexing."""
        if isinstance(item, int):
            if item < 0 or item >= self.shape[0]:
                raise IndexError("Index out of bounds")
            if self.ndim == 1:
                return self._values.get(item, self.dtype(0))
            elif item not in self._values:
                sub_shape = self.shape[1:]
                sub_array = nlp_array(sub_shape, dtype=self.dtype)
                self._values[item] = sub_array
            return self._values[item]

        elif isinstance(item, tuple):
            a = self[item[0]]
            if len(item) > 1:
                return a[item[1:]]
            return a

        raise ValueError(f"Unsupported item type: {type(item)}")

    def __setitem__(self, item, value):
        """N-D sparse array setter operation."""
        if isinstance(item, int):
            if self.ndim > 1:
                raise IndexError("Too few indices for array")
            if value != 0:
                self._values[item] = self.dtype(value)

        elif isinstance(item, tuple):
            if len(item) > self.ndim:
                raise IndexError("Too many indices for array")
            if len(item) < self.ndim:
                raise IndexError("Too few indices for array")
            if len(item) == 1:
                self[item[0]] = value
            else:
                self[item[0]][item[1:]] = value

        return self[item]

    def toarray(self):
        """Convert to a dense numpy array."""
        arr = np.zeros(self.shape, dtype=self.dtype)
        for idx, val in self._values.items():
            if self.ndim == 1:
                arr[idx] = val
            else:
                arr[idx] = val.toarray()
        return arr
