import numpy as np
from scipy.sparse import coo_matrix # Used for tocoo() method for 2D compatibility

class NDArrayCOO:
    """
    N-dimensional sparse array in Coordinate (COO) format.

    This class provides an N-dimensional sparse array that mimics some of the
    basic interface of scipy.sparse matrices, particularly the COO format,
    but extended to N dimensions. It stores non-zero elements by their
    coordinates and values.
    """

    def __init__(self, coords, data, shape, sum_duplicates=True):
        """
        Initialize the N-dimensional sparse array.

        Parameters
        ----------
        coords : array_like
            Coordinates of the non-zero elements.
            Expected shape is (nnz, ndim), where nnz is the number of
            non-zero elements and ndim is the number of dimensions.
        data : array_like
            Non-zero values corresponding to the coordinates.
            Expected shape is (nnz,).
        shape : tuple
            Shape of the N-dimensional array. Each element must be a
            non-negative integer.
        sum_duplicates : bool, optional
            If True, duplicate coordinates will have their corresponding
            data values summed upon initialization. Defaults to True.
        """
        if not isinstance(shape, tuple):
            raise TypeError("Shape must be a tuple.")
        if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            raise ValueError("Shape dimensions must be non-negative integers.")

        coords = np.asarray(coords)
        data = np.asarray(data)

        # If coords is empty, ensure it has the correct 2D shape (0, ndim)
        if coords.size == 0 and coords.ndim == 1:
            coords = coords.reshape(0, len(shape))

        if coords.ndim != 2:
            raise ValueError("Coords must be a 2D array (nnz, ndim).")
        if data.ndim != 1:
            raise ValueError("Data must be a 1D array.")
        if coords.shape[0] != data.shape[0]:
            raise ValueError("Number of coordinates must match number of data points.")
        if coords.shape[1] != len(shape):
            raise ValueError("Number of dimensions in coords must match length of shape.")

        # Validate coordinates are within bounds
        for i, dim_size in enumerate(shape):
            # Only check bounds if there are actual coordinates
            if coords.shape[0] > 0 and not np.all((coords[:, i] >= 0) & (coords[:, i] < dim_size)):
                raise ValueError(f"Coordinates for dimension {i} are out of bounds.")

        self._shape = tuple(shape)
        self._ndim = len(shape)

        if sum_duplicates:
            if coords.shape[0] == 0:
                self._coords = np.empty((0, self._ndim), dtype=coords.dtype)
                self._data = np.empty((0,), dtype=data.dtype)
                self._nnz = 0
            else:
                # Sort coordinates and data together based on the canonical order
                # (lexicographical sort by last dimension first, then second last, etc.)
                order = np.lexsort(coords.T)
                coords_sorted = coords[order]
                data_sorted = data[order]

                # Identify unique rows and sum their corresponding data
                # This approach avoids re-sorting by np.unique and preserves lexsort order
                unique_mask = np.ones(len(coords_sorted), dtype=bool)
                if len(coords_sorted) > 1:
                    unique_mask[1:] = np.any(coords_sorted[1:] != coords_sorted[:-1], axis=1)

                self._coords = coords_sorted[unique_mask]
                
                # Find the indices where unique coordinates start
                unique_indices = np.where(unique_mask)[0]
                
                new_data = np.zeros(len(unique_indices), dtype=data_sorted.dtype)
                
                # Iterate through unique blocks and sum their data
                for i, start_idx in enumerate(unique_indices):
                    end_idx = unique_indices[i+1] if i+1 < len(unique_indices) else len(data_sorted)
                    new_data[i] = np.sum(data_sorted[start_idx:end_idx])

                self._data = new_data
                self._nnz = len(new_data)
        else:
            self._coords = coords
            self._data = data
            self._nnz = data.shape[0]

    @property
    def coords(self):
        """The coordinates of the non-zero elements."""
        return self._coords

    @property
    def data(self):
        """The non-zero values."""
        return self._data

    @property
    def shape(self):
        """The shape of the array."""
        return self._shape

    @property
    def ndim(self):
        """The number of dimensions of the array."""
        return self._ndim

    @property
    def nnz(self):
        """The number of stored non-zero elements."""
        return self._nnz

    def todense(self):
        """
        Convert the sparse array to a dense NumPy array.

        Returns
        -------
        numpy.ndarray
            A dense array with the same shape and values.
        """
        dense_array = np.zeros(self.shape, dtype=self.data.dtype)
        # Use advanced indexing to place data at coordinates.
        # This works for N-dimensions by unpacking the transposed coordinates.
        # Only attempt to assign if there are non-zero elements
        if self.nnz > 0:
            dense_array[tuple(self.coords.T)] = self.data
        return dense_array

    def transpose(self, axes=None):
        """
        Transpose the sparse array by permuting its dimensions.

        Parameters
        ----------
        axes : tuple of ints, optional
            By default (None), reverses the dimensions. Otherwise, permute the axes
            according to the values given. The tuple must be a permutation of
            (0, 1, ..., ndim-1).

        Returns
        -------
        NDArrayCOO
            A new NDArrayCOO object with transposed dimensions.
        """
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        elif not isinstance(axes, tuple) or len(axes) != self.ndim or \
             set(axes) != set(range(self.ndim)):
            raise ValueError(f"Axes must be a permutation of (0, ..., {self.ndim-1}).")

        # Permute coordinates based on the new axis order
        new_coords = self.coords[:, axes]

        # Permute shape based on the new axis order
        new_shape = tuple(self.shape[ax] for ax in axes)

        # The new transposed array should also sum duplicates and thus sort its coordinates
        return NDArrayCOO(new_coords, self.data.copy(), new_shape, sum_duplicates=True)

    @property
    def T(self):
        """
        The transposed array.

        For 2D arrays, this is equivalent to `transpose()`.
        For N-D arrays, this reverses the dimensions.
        """
        return self.transpose()

    def __mul__(self, other):
        """Scalar multiplication."""
        if np.isscalar(other):
            return NDArrayCOO(self.coords.copy(), self.data * other, self.shape, sum_duplicates=False)
        raise NotImplementedError("Only scalar multiplication is supported for now.")

    def __rmul__(self, other):
        """Right scalar multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Scalar division."""
        if np.isscalar(other):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return NDArrayCOO(self.coords.copy(), self.data / other, self.shape, sum_duplicates=False)
        raise NotImplementedError("Only scalar division is supported for now.")

    def __add__(self, other):
        """
        Addition with another NDArrayCOO or a scalar.

        Note: Adding a non-zero scalar or a dense array will typically result
        in a dense array, as it changes all implicit zero values.
        """
        if isinstance(other, NDArrayCOO):
            if self.shape != other.shape:
                raise ValueError("Shapes must match for addition of two sparse arrays.")
            # For simplicity, this only handles addition where coordinates are identical.
            # A more robust implementation would merge coordinates and sum values.
            # For now, convert to dense if structures are not identical.
            if np.array_equal(self.coords, other.coords):
                return NDArrayCOO(self.coords.copy(), self.data + other.data, self.shape, sum_duplicates=False)
            else:
                # Fallback to dense addition if structures differ
                return self.todense() + other.todense()
        elif np.isscalar(other):
            if other == 0:
                return self
            else:
                # Adding a non-zero scalar makes the array dense
                return self.todense() + other
        elif isinstance(other, np.ndarray):
            if self.shape != other.shape:
                raise ValueError("Shapes must match for addition with a dense array.")
            return self.todense() + other
        raise NotImplementedError("Addition with other types or complex sparse structures is not fully implemented.")

    def __sub__(self, other):
        """
        Subtraction with another NDArrayCOO or a scalar.

        Note: Subtracting a non-zero scalar or a dense array will typically result
        in a dense array, as it changes all implicit zero values.
        """
        if isinstance(other, NDArrayCOO):
            if self.shape != other.shape:
                raise ValueError("Shapes must match for subtraction of two sparse arrays.")
            if np.array_equal(self.coords, other.coords):
                return NDArrayCOO(self.coords.copy(), self.data - other.data, self.shape, sum_duplicates=False)
            else:
                return self.todense() - other.todense()
        elif np.isscalar(other):
            if other == 0:
                return self
            else:
                return self.todense() - other
        elif isinstance(other, np.ndarray):
            if self.shape != other.shape:
                raise ValueError("Shapes must match for subtraction with a dense array.")
            return self.todense() - other
        raise NotImplementedError("Subtraction with other types or complex sparse structures is not fully implemented.")

    def __repr__(self):
        """Return a string representation of the object."""
        return (f"<NDArrayCOO: shape={self.shape}, nnz={self.nnz}, ndim={self.ndim}>\n"
                f"  coords={self.coords}\n"
                f"  data={self.data}")

    def __str__(self):
        """Return a string representation of the object."""
        return self.__repr__()

    # SciPy sparse matrix compatibility attributes (for 2D cases)
    @property
    def row(self):
        """Row indices (for 2D arrays). Raises AttributeError if not 2D."""
        if self.ndim != 2:
            raise AttributeError("`row` attribute is only available for 2D arrays.")
        return self.coords[:, 0]

    @property
    def col(self):
        """Column indices (for 2D arrays). Raises AttributeError if not 2D."""
        if self.ndim != 2:
            raise AttributeError("`col` attribute is only available for 2D arrays.")
        return self.coords[:, 1]

    def tocoo(self):
        """
        Convert to a scipy.sparse.coo_matrix if the array is 2-dimensional.

        Returns
        -------
        scipy.sparse.coo_matrix
            A 2D sparse matrix.

        Raises
        ------
        ValueError
            If the array is not 2-dimensional.
        """
        if self.ndim != 2:
            raise ValueError("Can only convert to scipy.sparse.coo_matrix for 2-dimensional arrays.")
        return coo_matrix((self.data, (self.coords[:, 0], self.coords[:, 1])), shape=self.shape)
