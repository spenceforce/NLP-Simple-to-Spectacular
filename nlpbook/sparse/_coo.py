import numpy as np
from scipy.sparse import coo_array as coo


class coo_array(coo):
    """
    Extension of `scipy.sparse.coo_array` with support for 1-D an 2-D slicing.
    """

    def __getitem__(self, key):
        """N-D coo array slicing."""
        if not self.has_canonical_format:
            self.sum_duplicates()

        if isinstance(key, tuple):
            a = self[key[0]]
            if len(key) > 1:
                return a[key[1:]]
            return a

        elif isinstance(key, int):
            i = np.nonzero(self.coords[0] == key)[0]
            if self.ndim == 1:
                if i.shape[0] == 0:
                    return np.array(0, dtype=self.dtype)
                return self.data[i[0]]

            elif i.shape[0] == 0:
                return type(self)(self.shape[1:], dtype=self.dtype)

            data = self.data[i]
            coords = tuple(self.coords[j][i] for j in range(1, len(self.coords)))
            return type(self)(
                (data, coords),
                shape=self.shape[1:],
                dtype=self.dtype,
            )

        elif isinstance(key, slice):
            i = np.nonzero(np.isin(self.coords[0], list(range(*key.indices(len(self.coords[0]))))))[0]
            data = self.data[i]
            coords = tuple(self.coords[j][i] for j in range(len(self.coords)))
            shape = (len(coords), *self.shape[1:])
            return type(self)((data, coords), shape=shape, dtype=self.dtype)

        raise ValueError(f"Unsupported key type: {type(key)}")
