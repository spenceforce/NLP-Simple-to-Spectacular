import numpy as np
import pytest

from nlpbook.sparse.nd_array import nd_array


def test_instantiate_1d_array():
    """Test instantiation of a 1D array."""
    dense_arr = np.array([1, 0, 2, 0, 3])
    sparse_arr = nd_array(dense_arr)
    assert isinstance(sparse_arr, nd_array)
    assert sparse_arr.shape == (5,)
    assert np.array_equal(sparse_arr.toarray(), dense_arr)


def test_instantiate_2d_array():
    """Test instantiation of a 2D array."""
    dense_arr = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    sparse_arr = nd_array(dense_arr)
    assert isinstance(sparse_arr, nd_array)
    assert sparse_arr.shape == (3, 3)
    assert np.array_equal(sparse_arr.toarray(), dense_arr)


def test_instantiate_3d_array():
    """Test instantiation of a 3D array."""
    dense_arr = np.zeros((2, 3, 4))
    dense_arr[0, 1, 2] = 5
    dense_arr[1, 0, 3] = 10
    sparse_arr = nd_array(dense_arr)
    assert isinstance(sparse_arr, nd_array)
    assert sparse_arr.shape == (2, 3, 4)
    assert np.array_equal(sparse_arr.toarray(), dense_arr)


def test_1d_indexing_read():
    """Test reading elements from a 1D sparse array."""
    dense_arr = np.array([10, 0, 20, 0, 30])
    sparse_arr = nd_array(dense_arr)

    # Single element access
    assert sparse_arr[0] == 10
    assert sparse_arr[1] == 0
    assert sparse_arr[4] == 30

    # Slice access
    assert np.array_equal(sparse_arr[1:4].toarray(), dense_arr[1:4])
    assert np.array_equal(sparse_arr[:].toarray(), dense_arr)


def test_2d_indexing_read():
    """Test reading elements from a 2D sparse array."""
    dense_arr = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    sparse_arr = nd_array(dense_arr)

    # Single element access
    assert sparse_arr[0, 0] == 1
    assert sparse_arr[1, 1] == 3
    assert sparse_arr[2, 2] == 5
    assert sparse_arr[0, 1] == 0

    # Row slice
    assert np.array_equal(sparse_arr[0, :].toarray(), dense_arr[0, :])
    # Column slice
    assert np.array_equal(sparse_arr[:, 1].toarray(), dense_arr[:, 1])
    # Sub-matrix slice
    assert np.array_equal(sparse_arr[0:2, 0:2].toarray(), dense_arr[0:2, 0:2])


def test_3d_indexing_read():
    """Test reading elements from a 3D sparse array."""
    dense_arr = np.zeros((2, 3, 4))
    dense_arr[0, 1, 2] = 5
    dense_arr[1, 0, 3] = 10
    sparse_arr = nd_array(dense_arr)

    # Single element access
    assert sparse_arr[0, 1, 2] == 5
    assert sparse_arr[1, 0, 3] == 10
    assert sparse_arr[0, 0, 0] == 0

    # Slice access (various combinations)
    assert np.array_equal(sparse_arr[0, :, :].toarray(), dense_arr[0, :, :])
    assert np.array_equal(sparse_arr[:, 1, :].toarray(), dense_arr[:, 1, :])
    assert np.array_equal(sparse_arr[:, :, 2].toarray(), dense_arr[:, :, 2])
    assert np.array_equal(sparse_arr[0, 0:2, 1:3].toarray(), dense_arr[0, 0:2, 1:3])


def test_1d_indexing_write():
    """Test writing elements to a 1D sparse array."""
    dense_arr = np.array([10, 0, 20, 0, 30])
    sparse_arr = nd_array(dense_arr)

    # Set single element
    sparse_arr[1] = 99
    assert sparse_arr[1] == 99
    assert np.array_equal(sparse_arr.toarray(), np.array([10, 99, 20, 0, 30]))

    # Set slice
    sparse_arr[2:4] = np.array([100, 101])
    assert np.array_equal(sparse_arr.toarray(), np.array([10, 99, 100, 101, 30]))


def test_2d_indexing_write():
    """Test writing elements to a 2D sparse array."""
    dense_arr = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    sparse_arr = nd_array(dense_arr)

    # Set single element
    sparse_arr[0, 1] = 99
    assert sparse_arr[0, 1] == 99
    expected_arr = np.array([[1, 99, 2], [0, 3, 0], [4, 0, 5]])
    assert np.array_equal(sparse_arr.toarray(), expected_arr)

    # Set row slice
    sparse_arr[1, :] = np.array([10, 20, 30])
    expected_arr = np.array([[1, 99, 2], [10, 20, 30], [4, 0, 5]])
    assert np.array_equal(sparse_arr.toarray(), expected_arr)

    # Set column slice
    sparse_arr[:, 2] = np.array([100, 200, 300])
    expected_arr = np.array([[1, 99, 100], [10, 20, 200], [4, 0, 300]])
    assert np.array_equal(sparse_arr.toarray(), expected_arr)

    # Set sub-matrix slice
    sparse_arr[0:2, 0:2] = np.array([[11, 22], [33, 44]])
    expected_arr = np.array([[11, 22, 100], [33, 44, 200], [4, 0, 300]])
    assert np.array_equal(sparse_arr.toarray(), expected_arr)


def test_3d_indexing_write():
    """Test writing elements to a 3D sparse array."""
    dense_arr = np.zeros((2, 3, 4))
    dense_arr[0, 1, 2] = 5
    sparse_arr = nd_array(dense_arr)

    # Set single element
    sparse_arr[0, 0, 0] = 1
    assert sparse_arr[0, 0, 0] == 1
    expected_arr = np.copy(dense_arr)
    expected_arr[0, 0, 0] = 1
    assert np.array_equal(sparse_arr.toarray(), expected_arr)

    # Set slice (e.g., a 2D plane)
    sparse_arr[1, :, :] = np.array([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
    expected_arr[1, :, :] = np.array([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
    assert np.array_equal(sparse_arr.toarray(), expected_arr)

    # Set sub-slice
    sparse_arr[0, 0:2, 0:2] = np.array([[100, 101], [102, 103]])
    expected_arr[0, 0:2, 0:2] = np.array([[100, 101], [102, 103]])
    assert np.array_equal(sparse_arr.toarray(), expected_arr)
