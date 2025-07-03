import numpy as np
import pytest
from scipy.sparse import coo_matrix

from nlpbook.sparse.nd_array import NDArrayCOO


def test_init_valid_2d():
    coords = [[0, 0], [1, 1], [0, 1]]
    data = [1, 2, 3]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    assert arr.shape == shape
    assert arr.ndim == 2
    assert arr.nnz == 3  # Duplicates are summed by default
    np.testing.assert_array_equal(arr.coords, [[0, 0], [0, 1], [1, 1]])
    np.testing.assert_array_equal(arr.data, [1, 3, 2])


def test_init_valid_3d():
    coords = [[0, 0, 0], [1, 1, 1], [0, 1, 0]]
    data = [1, 2, 3]
    shape = (2, 2, 2)
    arr = NDArrayCOO(coords, data, shape)
    assert arr.shape == shape
    assert arr.ndim == 3
    assert arr.nnz == 3
    np.testing.assert_array_equal(arr.coords, [[0, 0, 0], [0, 1, 0], [1, 1, 1]])
    np.testing.assert_array_equal(arr.data, [1, 3, 2])


def test_init_sum_duplicates_true():
    coords = [[0, 0], [1, 1], [0, 0]]
    data = [1, 2, 3]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape, sum_duplicates=True)
    assert arr.nnz == 2
    np.testing.assert_array_equal(arr.coords, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(arr.data, [4, 2])


def test_init_sum_duplicates_false():
    coords = [[0, 0], [1, 1], [0, 0]]
    data = [1, 2, 3]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape, sum_duplicates=False)
    assert arr.nnz == 3
    # Order might change due to internal asarray conversion, but values should be there
    expected_coords = np.array([[0, 0], [1, 1], [0, 0]])
    expected_data = np.array([1, 2, 3])
    # Check if all elements are present, order doesn't matter for sum_duplicates=False
    assert len(arr.coords) == len(expected_coords)
    assert len(arr.data) == len(expected_data)
    for c, d in zip(expected_coords, expected_data):
        assert any(np.array_equal(c, arr_c) and d == arr_d for arr_c, arr_d in zip(arr.coords, arr.data))


def test_init_empty():
    coords = []
    data = []
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    assert arr.shape == shape
    assert arr.ndim == 2
    assert arr.nnz == 0
    assert arr.coords.shape == (0, 2)
    assert arr.data.shape == (0,)


def test_init_invalid_shape_type():
    coords = [[0, 0]]
    data = [1]
    with pytest.raises(TypeError, match="Shape must be a tuple."):
        NDArrayCOO(coords, data, [2, 2])


def test_init_invalid_shape_value():
    coords = [[0, 0]]
    data = [1]
    with pytest.raises(ValueError, match="Shape dimensions must be non-negative integers."):
        NDArrayCOO(coords, data, (2, -1))


def test_init_coords_ndim_error():
    coords = [0, 0]  # 1D
    data = [1]
    shape = (2, 2)
    with pytest.raises(ValueError, match="Coords must be a 2D array"):
        NDArrayCOO(coords, data, shape)


def test_init_data_ndim_error():
    coords = [[0, 0]]
    data = [[1]]  # 2D
    shape = (2, 2)
    with pytest.raises(ValueError, match="Data must be a 1D array."):
        NDArrayCOO(coords, data, shape)


def test_init_coords_data_mismatch():
    coords = [[0, 0], [1, 1]]
    data = [1]
    shape = (2, 2)
    with pytest.raises(ValueError, match="Number of coordinates must match number of data points."):
        NDArrayCOO(coords, data, shape)


def test_init_coords_shape_ndim_mismatch():
    coords = [[0, 0, 0]]  # 3D coords
    data = [1]
    shape = (2, 2)  # 2D shape
    with pytest.raises(ValueError, match="Number of dimensions in coords must match length of shape."):
        NDArrayCOO(coords, data, shape)


def test_init_coords_out_of_bounds():
    coords = [[0, 0], [2, 0]]  # [2,0] is out of bounds for shape (2,2)
    data = [1, 2]
    shape = (2, 2)
    with pytest.raises(ValueError, match="Coordinates for dimension 0 are out of bounds."):
        NDArrayCOO(coords, data, shape)


def test_properties():
    coords = np.array([[0, 0], [1, 1]])
    data = np.array([1, 2])
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape, sum_duplicates=False)
    np.testing.assert_array_equal(arr.coords, coords)
    np.testing.assert_array_equal(arr.data, data)
    assert arr.shape == shape
    assert arr.ndim == 2
    assert arr.nnz == 2


def test_todense_2d():
    coords = [[0, 0], [1, 1], [0, 1]]
    data = [1, 2, 3]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    expected_dense = np.array([[1, 3], [0, 2]])
    np.testing.assert_array_equal(arr.todense(), expected_dense)


def test_todense_3d():
    coords = [[0, 0, 0], [1, 1, 1], [0, 1, 0]]
    data = [1, 2, 3]
    shape = (2, 2, 2)
    arr = NDArrayCOO(coords, data, shape)
    expected_dense = np.zeros(shape)
    expected_dense[0, 0, 0] = 1
    expected_dense[0, 1, 0] = 3
    expected_dense[1, 1, 1] = 2
    np.testing.assert_array_equal(arr.todense(), expected_dense)


def test_todense_empty():
    coords = []
    data = []
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    expected_dense = np.zeros(shape)
    np.testing.assert_array_equal(arr.todense(), expected_dense)


def test_transpose_2d_default():
    coords = [[0, 1], [1, 0]]
    data = [10, 20]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    transposed_arr = arr.transpose()
    assert transposed_arr.shape == (2, 2)
    np.testing.assert_array_equal(transposed_arr.coords, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(transposed_arr.data, [20, 10])
    np.testing.assert_array_equal(transposed_arr.todense(), arr.todense().T)


def test_transpose_3d_default():
    coords = [[0, 1, 2], [1, 0, 1]]
    data = [10, 20]
    shape = (2, 2, 3)
    arr = NDArrayCOO(coords, data, shape)
    transposed_arr = arr.transpose()
    assert transposed_arr.shape == (3, 2, 2)
    np.testing.assert_array_equal(transposed_arr.coords, [[2, 1, 0], [1, 0, 1]])
    np.testing.assert_array_equal(transposed_arr.data, [10, 20])
    np.testing.assert_array_equal(transposed_arr.todense(), arr.todense().T)


def test_transpose_3d_specified_axes():
    coords = [[0, 1, 2], [1, 0, 1]]
    data = [10, 20]
    shape = (2, 2, 3)
    arr = NDArrayCOO(coords, data, shape)
    transposed_arr = arr.transpose(axes=(1, 2, 0))
    assert transposed_arr.shape == (2, 3, 2)
    np.testing.assert_array_equal(transposed_arr.coords, [[1, 2, 0], [0, 1, 1]])
    np.testing.assert_array_equal(transposed_arr.data, [10, 20])
    np.testing.assert_array_equal(transposed_arr.todense(), np.transpose(arr.todense(), axes=(1, 2, 0)))


def test_transpose_invalid_axes():
    coords = [[0, 0]]
    data = [1]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    with pytest.raises(ValueError, match="Axes must be a permutation of"):
        arr.transpose(axes=(0, 2))  # Invalid length
    with pytest.raises(ValueError, match="Axes must be a permutation of"):
        arr.transpose(axes=(0, 0))  # Not a permutation


def test_T_property():
    coords = [[0, 1], [1, 0]]
    data = [10, 20]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    assert arr.T.shape == (2, 2)
    np.testing.assert_array_equal(arr.T.todense(), arr.todense().T)


def test_mul_scalar():
    coords = [[0, 0], [1, 1]]
    data = [1, 2]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    result = arr * 5
    np.testing.assert_array_equal(result.coords, arr.coords)
    np.testing.assert_array_equal(result.data, [5, 10])
    np.testing.assert_array_equal(result.todense(), arr.todense() * 5)


def test_rmul_scalar():
    coords = [[0, 0], [1, 1]]
    data = [1, 2]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    result = 5 * arr
    np.testing.assert_array_equal(result.coords, arr.coords)
    np.testing.assert_array_equal(result.data, [5, 10])
    np.testing.assert_array_equal(result.todense(), 5 * arr.todense())


def test_truediv_scalar():
    coords = [[0, 0], [1, 1]]
    data = [10, 20]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    result = arr / 5
    np.testing.assert_array_equal(result.coords, arr.coords)
    np.testing.assert_array_equal(result.data, [2, 4])
    np.testing.assert_array_equal(result.todense(), arr.todense() / 5)


def test_truediv_by_zero():
    coords = [[0, 0]]
    data = [1]
    shape = (1, 1)
    arr = NDArrayCOO(coords, data, shape)
    with pytest.raises(ZeroDivisionError, match="Cannot divide by zero."):
        arr / 0


def test_add_ndarraycoo_identical_coords():
    coords = [[0, 0], [1, 1]]
    data1 = [1, 2]
    data2 = [3, 4]
    shape = (2, 2)
    arr1 = NDArrayCOO(coords, data1, shape)
    arr2 = NDArrayCOO(coords, data2, shape)
    result = arr1 + arr2
    np.testing.assert_array_equal(result.coords, arr1.coords)
    np.testing.assert_array_equal(result.data, [4, 6])
    np.testing.assert_array_equal(result.todense(), arr1.todense() + arr2.todense())


def test_add_ndarraycoo_different_coords_fallback_to_dense():
    coords1 = [[0, 0], [1, 1]]
    data1 = [1, 2]
    coords2 = [[0, 1], [1, 1]]
    data2 = [3, 4]
    shape = (2, 2)
    arr1 = NDArrayCOO(coords1, data1, shape)
    arr2 = NDArrayCOO(coords2, data2, shape)
    result = arr1 + arr2
    expected_dense = np.array([[1, 3], [0, 6]])
    np.testing.assert_array_equal(result, expected_dense) # Result is dense array


def test_add_ndarraycoo_shape_mismatch():
    coords1 = [[0, 0]]
    data1 = [1]
    shape1 = (2, 2)
    coords2 = [[0, 0]]
    data2 = [1]
    shape2 = (3, 3)
    arr1 = NDArrayCOO(coords1, data1, shape1)
    arr2 = NDArrayCOO(coords2, data2, shape2)
    with pytest.raises(ValueError, match="Shapes must match for addition of two sparse arrays."):
        arr1 + arr2


def test_add_scalar_zero():
    coords = [[0, 0]]
    data = [1]
    shape = (1, 1)
    arr = NDArrayCOO(coords, data, shape)
    result = arr + 0
    assert result is arr # Should return self


def test_add_scalar_nonzero():
    coords = [[0, 0]]
    data = [1]
    shape = (1, 1)
    arr = NDArrayCOO(coords, data, shape)
    result = arr + 5
    expected_dense = np.array([[6]])
    np.testing.assert_array_equal(result, expected_dense) # Result is dense array


def test_add_dense_array():
    coords = [[0, 0], [1, 1]]
    data = [1, 2]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    dense_arr = np.array([[10, 20], [30, 40]])
    result = arr + dense_arr
    expected_dense = np.array([[11, 20], [30, 42]])
    np.testing.assert_array_equal(result, expected_dense)


def test_add_dense_array_shape_mismatch():
    coords = [[0, 0]]
    data = [1]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    dense_arr = np.array([1, 2])
    with pytest.raises(ValueError, match="Shapes must match for addition with a dense array."):
        arr + dense_arr


def test_sub_ndarraycoo_identical_coords():
    coords = [[0, 0], [1, 1]]
    data1 = [5, 6]
    data2 = [1, 2]
    shape = (2, 2)
    arr1 = NDArrayCOO(coords, data1, shape)
    arr2 = NDArrayCOO(coords, data2, shape)
    result = arr1 - arr2
    np.testing.assert_array_equal(result.coords, arr1.coords)
    np.testing.assert_array_equal(result.data, [4, 4])
    np.testing.assert_array_equal(result.todense(), arr1.todense() - arr2.todense())


def test_sub_ndarraycoo_different_coords_fallback_to_dense():
    coords1 = [[0, 0], [1, 1]]
    data1 = [5, 6]
    coords2 = [[0, 1], [1, 1]]
    data2 = [1, 2]
    shape = (2, 2)
    arr1 = NDArrayCOO(coords1, data1, shape)
    arr2 = NDArrayCOO(coords2, data2, shape)
    result = arr1 - arr2
    expected_dense = np.array([[5, -1], [0, 4]])
    np.testing.assert_array_equal(result, expected_dense) # Result is dense array


def test_sub_ndarraycoo_shape_mismatch():
    coords1 = [[0, 0]]
    data1 = [1]
    shape1 = (2, 2)
    coords2 = [[0, 0]]
    data2 = [1]
    shape2 = (3, 3)
    arr1 = NDArrayCOO(coords1, data1, shape1)
    arr2 = NDArrayCOO(coords2, data2, shape2)
    with pytest.raises(ValueError, match="Shapes must match for subtraction of two sparse arrays."):
        arr1 - arr2


def test_sub_scalar_zero():
    coords = [[0, 0]]
    data = [1]
    shape = (1, 1)
    arr = NDArrayCOO(coords, data, shape)
    result = arr - 0
    assert result is arr # Should return self


def test_sub_scalar_nonzero():
    coords = [[0, 0]]
    data = [1]
    shape = (1, 1)
    arr = NDArrayCOO(coords, data, shape)
    result = arr - 5
    expected_dense = np.array([[-4]])
    np.testing.assert_array_equal(result, expected_dense) # Result is dense array


def test_sub_dense_array():
    coords = [[0, 0], [1, 1]]
    data = [1, 2]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    dense_arr = np.array([[10, 20], [30, 40]])
    result = arr - dense_arr
    expected_dense = np.array([[-9, -20], [-30, -38]])
    np.testing.assert_array_equal(result, expected_dense)


def test_sub_dense_array_shape_mismatch():
    coords = [[0, 0]]
    data = [1]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    dense_arr = np.array([1, 2])
    with pytest.raises(ValueError, match="Shapes must match for subtraction with a dense array."):
        arr - dense_arr


def test_row_col_properties_2d():
    coords = [[0, 1], [1, 0]]
    data = [10, 20]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    np.testing.assert_array_equal(arr.row, [0, 1])
    np.testing.assert_array_equal(arr.col, [1, 0])


def test_row_col_properties_nd_raises_error():
    coords = [[0, 1, 2]]
    data = [10]
    shape = (2, 2, 3)
    arr = NDArrayCOO(coords, data, shape)
    with pytest.raises(AttributeError, match="`row` attribute is only available for 2D arrays."):
        arr.row
    with pytest.raises(AttributeError, match="`col` attribute is only available for 2D arrays."):
        arr.col


def test_tocoo_2d():
    coords = [[0, 1], [1, 0]]
    data = [10, 20]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    scipy_coo = arr.tocoo()
    assert isinstance(scipy_coo, coo_matrix)
    np.testing.assert_array_equal(scipy_coo.row, arr.row)
    np.testing.assert_array_equal(scipy_coo.col, arr.col)
    np.testing.assert_array_equal(scipy_coo.data, arr.data)
    assert scipy_coo.shape == arr.shape


def test_tocoo_nd_raises_error():
    coords = [[0, 1, 2]]
    data = [10]
    shape = (2, 2, 3)
    arr = NDArrayCOO(coords, data, shape)
    with pytest.raises(ValueError, match="Can only convert to scipy.sparse.coo_matrix for 2-dimensional arrays."):
        arr.tocoo()


def test_repr_str():
    coords = [[0, 0], [1, 1]]
    data = [1, 2]
    shape = (2, 2)
    arr = NDArrayCOO(coords, data, shape)
    repr_str = repr(arr)
    assert "NDArrayCOO" in repr_str
    assert "shape=(2, 2)" in repr_str
    assert "nnz=2" in repr_str
    assert "ndim=2" in repr_str
    assert "coords=" in repr_str
    assert "data=" in repr_str
    assert str(arr) == repr_str
