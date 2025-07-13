import numpy as np
import pytest
from scipy.sparse import coo_array as sp_coo_array
from nlpbook.sparse._coo import coo_array


class TestCooArray:
    """
    Tests for the coo_array class, extending scipy.sparse.coo_array.
    These tests verify that slicing (including N-D) behaves as expected,
    ideally matching the behavior of scipy.sparse.coo_array.
    """

    def _create_2d_array(self):
        """Helper to create a sample 2D coo_array."""
        data = np.array([1, 2, 3, 4, 5, 6])
        row = np.array([0, 0, 1, 1, 2, 2])
        col = np.array([0, 1, 0, 1, 0, 1])
        shape = (3, 2)
        return coo_array((data, (row, col)), shape=shape)

    def _create_3d_array(self):
        """Helper to create a sample 3D coo_array."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        coords = np.array([
            [0, 0, 0, 0, 1, 1, 1, 1],  # dim 0
            [0, 0, 1, 1, 0, 0, 1, 1],  # dim 1
            [0, 1, 0, 1, 0, 1, 0, 1]   # dim 2
        ])
        shape = (2, 2, 2)
        return coo_array((data, coords), shape=shape)

    def test_single_element_access_2d(self):
        """Test accessing a single element in a 2D array."""
        arr = self._create_2d_array()
        assert arr[0, 0] == 1
        assert arr[1, 1] == 4
        assert arr[2, 0] == 5
        assert arr[0, 2] == 0  # Non-existent element should return 0

    def test_single_element_access_3d(self):
        """Test accessing a single element in a 3D array."""
        arr = self._create_3d_array()
        assert arr[0, 0, 0] == 1
        assert arr[1, 1, 1] == 8
        assert arr[0, 1, 0] == 3
        assert arr[0, 0, 2] == 0  # Non-existent element should return 0

    def test_1d_slicing_rows_2d(self):
        """Test slicing a 2D array using a single slice (implies row slicing)."""
        arr = self._create_2d_array()
        sliced_arr = arr[0:2]
        expected_data = np.array([1, 2, 3, 4])
        expected_row = np.array([0, 0, 1, 1])
        expected_col = np.array([0, 1, 0, 1])
        expected_shape = (2, 2)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.row, expected_row)
        assert np.array_equal(sliced_arr.col, expected_col)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_fancy_indexing_rows_2d(self):
        """Test fancy indexing for rows in a 2D array."""
        arr = self._create_2d_array()
        sliced_arr = arr[[0, 2]]
        expected_data = np.array([1, 2, 5, 6])
        expected_row = np.array([0, 0, 1, 1])  # New row indices relative to the slice
        expected_col = np.array([0, 1, 0, 1])
        expected_shape = (2, 2)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.row, expected_row)
        assert np.array_equal(sliced_arr.col, expected_col)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_2d_slicing_row_slice_all_cols(self):
        """Test 2D slicing with a row slice and all columns."""
        arr = self._create_2d_array()
        sliced_arr = arr[0:2, :]
        expected_data = np.array([1, 2, 3, 4])
        expected_row = np.array([0, 0, 1, 1])
        expected_col = np.array([0, 1, 0, 1])
        expected_shape = (2, 2)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.row, expected_row)
        assert np.array_equal(sliced_arr.col, expected_col)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_2d_slicing_all_rows_col_slice(self):
        """Test 2D slicing with all rows and a column slice."""
        arr = self._create_2d_array()
        sliced_arr = arr[:, 0:1]
        expected_data = np.array([1, 3, 5])
        expected_row = np.array([0, 1, 2])
        expected_col = np.array([0, 0, 0])
        expected_shape = (3, 1)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.row, expected_row)
        assert np.array_equal(sliced_arr.col, expected_col)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_2d_slicing_both_slices(self):
        """Test 2D slicing with both row and column slices."""
        arr = self._create_2d_array()
        sliced_arr = arr[0:2, 0:1]
        expected_data = np.array([1, 3])
        expected_row = np.array([0, 1])
        expected_col = np.array([0, 0])
        expected_shape = (2, 1)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.row, expected_row)
        assert np.array_equal(sliced_arr.col, expected_col)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_2d_fancy_indexing_both_coords(self):
        """Test 2D fancy indexing for both rows and columns (coordinate-wise)."""
        arr = self._create_2d_array()
        # This selects elements at (0,0) and (2,1)
        sliced_arr = arr[[0, 2], [0, 1]]
        assert np.array_equal(sliced_arr, np.array([1, 6]))  # Scipy returns a 1D array for this type of fancy indexing

    def test_nd_slicing_one_dim(self):
        """Test N-D slicing by fixing one dimension."""
        arr = self._create_3d_array()
        sliced_arr = arr[0, :, :]
        expected_data = np.array([1, 2, 3, 4])
        expected_coords = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1]
        ])
        expected_shape = (2, 2)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.coords, expected_coords)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_nd_slicing_multiple_dims(self):
        """Test N-D slicing with multiple dimension slices."""
        arr = self._create_3d_array()
        sliced_arr = arr[0:1, 0:1, :]  # Equivalent to arr[0,0,:]
        expected_data = np.array([1, 2])
        expected_coords = np.array([
            [0, 0],
            [0, 1]
        ])
        expected_shape = (1, 2)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.coords, expected_coords)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_nd_fancy_indexing_reorder_dim(self):
        """Test N-D fancy indexing that reorders a dimension."""
        arr = self._create_3d_array()
        # Compare with dense for complex fancy indexing as sparse coordinate order might change
        assert np.array_equal(arr[[1, 0], :, :].todense(), arr.todense()[[1, 0], :, :])
        assert arr[[1, 0], :, :].shape == arr.shape
        assert isinstance(arr[[1, 0], :, :], coo_array)

    def test_return_type_scalar(self):
        """Test that single element access returns a scalar."""
        arr = self._create_2d_array()
        assert isinstance(arr[0, 0], (int, float, np.integer, np.floating))

    def test_empty_slice_2d(self):
        """Test slicing that results in an empty array."""
        arr = self._create_2d_array()
        sliced_arr = arr[10:, :]
        assert sliced_arr.nnz == 0
        assert sliced_arr.shape == (0, 2)
        assert isinstance(sliced_arr, coo_array)

    def test_negative_indexing_2d(self):
        """Test negative indexing for rows in a 2D array."""
        arr = self._create_2d_array()
        sliced_arr = arr[-1, :]
        expected_data = np.array([5, 6])
        expected_row = np.array([0, 0])
        expected_col = np.array([0, 1])
        expected_shape = (1, 2)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.row, expected_row)
        assert np.array_equal(sliced_arr.col, expected_col)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_step_slicing_2d(self):
        """Test step slicing for rows in a 2D array."""
        arr = self._create_2d_array()
        sliced_arr = arr[::2, :]
        expected_data = np.array([1, 2, 5, 6])
        expected_row = np.array([0, 0, 1, 1])
        expected_col = np.array([0, 1, 0, 1])
        expected_shape = (2, 2)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.row, expected_row)
        assert np.array_equal(sliced_arr.col, expected_col)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_mixed_indexing_3d(self):
        """Test mixed integer and slice indexing in a 3D array."""
        arr = self._create_3d_array()
        sliced_arr = arr[0, :, 1]
        expected_data = np.array([2, 4])
        expected_coords = np.array([
            [0, 1]
        ])
        expected_shape = (2,)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.coords, expected_coords)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_slicing_with_ellipsis_3d(self):
        """Test slicing using ellipsis in a 3D array."""
        arr = self._create_3d_array()
        sliced_arr = arr[..., 0]
        expected_data = np.array([1, 3, 5, 7])
        expected_coords = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1]
        ])
        expected_shape = (2, 2)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.coords, expected_coords)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_slicing_with_newaxis_2d(self):
        """Test slicing with np.newaxis to add a new dimension."""
        arr = self._create_2d_array()
        sliced_arr = arr[:, np.newaxis, :]
        expected_data = np.array([1, 2, 3, 4, 5, 6])
        expected_coords = np.array([
            [0, 0, 1, 1, 2, 2],  # original dim 0
            [0, 0, 0, 0, 0, 0],  # new axis
            [0, 1, 0, 1, 0, 1]   # original dim 1
        ])
        expected_shape = (3, 1, 2)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.coords, expected_coords)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_slicing_with_boolean_array_raises_error(self):
        """Test that boolean array indexing raises an IndexError."""
        arr = self._create_2d_array()
        boolean_mask = np.array([True, False, True])
        with pytest.raises(IndexError, match="Boolean array indexing is not supported"):
            _ = arr[boolean_mask, :]

    def test_slicing_with_invalid_key_type_raises_error(self):
        """Test that an invalid key type raises an IndexError."""
        arr = self._create_2d_array()
        with pytest.raises(IndexError):
            _ = arr["invalid"]

    def test_slicing_out_of_bounds_no_error(self):
        """Test that slicing out of bounds results in an empty array, not an error."""
        arr = self._create_2d_array()
        sliced_arr = arr[10:, :]
        assert sliced_arr.nnz == 0
        assert sliced_arr.shape == (0, 2)
        assert isinstance(sliced_arr, coo_array)

    def test_slicing_too_many_indices_raises_error(self):
        """Test that providing too many indices raises an IndexError."""
        arr = self._create_2d_array()
        with pytest.raises(IndexError, match="too many indices for array"):
            _ = arr[0, 0, 0]

    def test_slicing_reduces_dimensions_3d_to_2d(self):
        """Test slicing a 3D array that reduces it to 2D."""
        arr = self._create_3d_array()
        sliced_arr = arr[0]  # Should return a 2D array (the last two dimensions)
        expected_data = np.array([1, 2, 3, 4])
        expected_coords = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1]
        ])
        expected_shape = (2, 2)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.coords, expected_coords)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_slicing_reduces_dimensions_3d_to_1d(self):
        """Test slicing a 3D array that reduces it to 1D."""
        arr = self._create_3d_array()
        sliced_arr = arr[0, 0]  # Should return a 1D array (the last dimension)
        expected_data = np.array([1, 2])
        expected_coords = np.array([
            [0, 1]
        ])
        expected_shape = (2,)
        assert np.array_equal(sliced_arr.data, expected_data)
        assert np.array_equal(sliced_arr.coords, expected_coords)
        assert sliced_arr.shape == expected_shape
        assert isinstance(sliced_arr, coo_array)

    def test_comparison_with_scipy_coo_array_all_slicing_types(self):
        """
        Test that nlpbook.sparse.coo_array slicing behavior matches
        scipy.sparse.coo_array for various slicing types.
        """
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        coords = np.array([
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1]
        ])
        shape = (2, 2, 2)
        arr_nlp = coo_array((data, coords), shape=shape)
        arr_sp = sp_coo_array((data, coords), shape=shape)

        # Test simple slice
        slice_key = (slice(0, 1), slice(None), slice(None))  # arr[0, :, :]
        sliced_nlp = arr_nlp[slice_key]
        sliced_sp = arr_sp[slice_key]
        assert np.array_equal(sliced_nlp.data, sliced_sp.data)
        assert np.array_equal(sliced_nlp.coords, sliced_sp.coords)
        assert sliced_nlp.shape == sliced_sp.shape
        assert isinstance(sliced_nlp, coo_array)

        # Test integer indexing (reduces dimension)
        int_key = (0, slice(None), slice(None))  # arr[0, :, :]
        sliced_nlp = arr_nlp[int_key]
        sliced_sp = arr_sp[int_key]
        assert np.array_equal(sliced_nlp.data, sliced_sp.data)
        assert np.array_equal(sliced_nlp.coords, sliced_sp.coords)
        assert sliced_nlp.shape == sliced_sp.shape
        assert isinstance(sliced_nlp, coo_array)

        # Test mixed slice and integer
        mixed_key = (0, slice(None), 1)  # arr[0, :, 1]
        sliced_nlp = arr_nlp[mixed_key]
        sliced_sp = arr_sp[mixed_key]
        # Compare dense representations for robustness as sparse coordinate order might differ
        assert np.array_equal(sliced_nlp.todense(), sliced_sp.todense())
        assert sliced_nlp.shape == sliced_sp.shape
        # The type might change if it becomes a scalar or a dense array.
        if sliced_nlp.ndim > 0:
            assert isinstance(sliced_nlp, coo_array)
        else:
            assert isinstance(sliced_nlp, (int, float, np.integer, np.floating))

        # Test fancy indexing (compare dense for robustness)
        fancy_key = ([0, 1], slice(None), [0, 1])  # arr[[0,1], :, [0,1]]
        sliced_nlp = arr_nlp[fancy_key]
        sliced_sp = arr_sp[fancy_key]
        assert np.array_equal(sliced_nlp.todense(), sliced_sp.todense())
        assert sliced_nlp.shape == sliced_sp.shape
        assert isinstance(sliced_nlp, coo_array)

        # Test single element access
        assert arr_nlp[0, 0, 0] == arr_sp[0, 0, 0]
        assert arr_nlp[1, 1, 1] == arr_sp[1, 1, 1]
        assert arr_nlp[0, 0, 2] == arr_sp[0, 0, 2]  # Non-existent element

        # Test ellipsis
        ellipsis_key = (..., 0)
        sliced_nlp = arr_nlp[ellipsis_key]
        sliced_sp = arr_sp[ellipsis_key]
        assert np.array_equal(sliced_nlp.todense(), sliced_sp.todense())
        assert sliced_nlp.shape == sliced_sp.shape
        assert isinstance(sliced_nlp, coo_array)

        # Test newaxis
        newaxis_key = (np.newaxis, ...)
        sliced_nlp = arr_nlp[newaxis_key]
        sliced_sp = arr_sp[newaxis_key]
        assert np.array_equal(sliced_nlp.todense(), sliced_sp.todense())
        assert sliced_nlp.shape == sliced_sp.shape
        assert isinstance(sliced_nlp, coo_array)
