import numpy as np
import pytest

import osipi


def test_signal_linear_array_like_inputs():
    # Test with different array-like input types

    # NumPy array input
    R1 = np.array([0.0, 1.5, 3.0, 4.0, 10.0], dtype=np.float64)
    k = np.float64(150.0)
    S_truth = np.array([0.0, 225.0, 450.0, 600.0, 1500.0])
    S = osipi.signal_linear(R1, k)
    np.testing.assert_allclose(S_truth, S, rtol=0, atol=1e-7)

    # Python list input
    R1_list = [0.0, 1.5, 3.0, 4.0, 10.0]
    S = osipi.signal_linear(R1_list, k)
    np.testing.assert_allclose(S_truth, S, rtol=0, atol=1e-7)

    # Python tuple input
    R1_tuple = (0.0, 1.5, 3.0, 4.0, 10.0)
    S = osipi.signal_linear(R1_tuple, k)
    np.testing.assert_allclose(S_truth, S, rtol=0, atol=1e-7)

    # Test with float32 dtype
    R1_32 = np.array([0.0, 1.5, 3.0, 4.0, 10.0], dtype=np.float32)
    k_32 = np.float32(150.0)
    S = osipi.signal_linear(R1_32, k_32)
    np.testing.assert_allclose(S_truth, S, rtol=1e-6, atol=1e-5)  # Reduced precision for float32

    # Test with invalid input
    with pytest.raises(ValueError):
        osipi.signal_linear("invalid input", k)


def test_signal_SPGR_array_like_inputs():
    # Test with different array-like input types
    R1 = np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float64)
    S0 = np.float64(100)
    TR = np.float64(5e-3)
    a = np.float64(15)

    # Calculate expected output with numpy arrays
    S_truth = osipi.signal_SPGR(R1, S0, TR, a)

    # Python list input for R1
    R1_list = [0.1, 0.2, 0.5, 1.0]
    S = osipi.signal_SPGR(R1_list, S0, TR, a)
    np.testing.assert_allclose(S_truth, S, rtol=0, atol=1e-7)

    # Python tuple input for R1
    R1_tuple = (0.1, 0.2, 0.5, 1.0)
    S = osipi.signal_SPGR(R1_tuple, S0, TR, a)
    np.testing.assert_allclose(S_truth, S, rtol=0, atol=1e-7)

    # Test with array-like S0
    S0_list = [100, 100, 100, 100]
    S = osipi.signal_SPGR(R1, S0_list, TR, a)
    np.testing.assert_allclose(S_truth, S, rtol=0, atol=1e-7)

    # Test with float32 dtype
    R1_32 = np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float32)
    S0_32 = np.float32(100)
    TR_32 = np.float32(5e-3)
    a_32 = np.float32(15)
    S = osipi.signal_SPGR(R1_32, S0_32, TR_32, a_32)
    # Increase tolerance for float32 precision
    np.testing.assert_allclose(S_truth, S, rtol=1e-4, atol=1e-4)

    # Test with invalid input
    with pytest.raises(ValueError):
        osipi.signal_SPGR("invalid input", S0, TR, a)


def test_S_to_R1_SPGR_array_like_inputs():
    # Test data adapted from OSIPI repository tests
    S_array = np.array([7, 9, 6, 10], dtype=np.float64)
    S_baseline = np.float64(7)
    R10 = np.float64(1 / 1.4)
    TR = np.float64(0.002)
    a = np.float64(13)

    # Calculate expected result with numpy array
    R1_truth = osipi.S_to_R1_SPGR(S_array, S_baseline, R10, TR, a)

    # Test with Python list
    S_list = [7.0, 9.0, 6.0, 10.0]
    R1 = osipi.S_to_R1_SPGR(S_list, S_baseline, R10, TR, a)
    np.testing.assert_allclose(R1_truth, R1, rtol=0, atol=1e-7)

    # Test with Python tuple
    S_tuple = (7.0, 9.0, 6.0, 10.0)
    R1 = osipi.S_to_R1_SPGR(S_tuple, S_baseline, R10, TR, a)
    np.testing.assert_allclose(R1_truth, R1, rtol=0, atol=1e-7)

    # Test with float32 dtype
    S_32 = np.array([7, 9, 6, 10], dtype=np.float32)
    S_baseline_32 = np.float32(7)
    R10_32 = np.float32(1 / 1.4)
    TR_32 = np.float32(0.002)
    a_32 = np.float32(13)
    R1 = osipi.S_to_R1_SPGR(S_32, S_baseline_32, R10_32, TR_32, a_32)
    # Increase tolerance for float32 precision
    np.testing.assert_allclose(R1_truth, R1, rtol=1e-4, atol=1e-4)

    # Test with non-array input (should raise error as function expects 1D array)
    with pytest.raises(TypeError):
        osipi.S_to_R1_SPGR(42, S_baseline, R10, TR, a)

    # Test with 2D array (should raise error as function expects 1D array)
    with pytest.raises(TypeError):
        osipi.S_to_R1_SPGR(np.array([[1, 2], [3, 4]]), S_baseline, R10, TR, a)


def test_R1_to_C_linear_relaxivity_array_like_inputs():
    # Test with different array-like input types

    # NumPy array input
    R1_array = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    R10 = np.float64(1)
    r1 = np.float64(5)
    C_truth = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    C = osipi.R1_to_C_linear_relaxivity(R1_array, R10, r1)
    np.testing.assert_allclose(C_truth, C, rtol=0, atol=1e-7)

    # Python list input
    R1_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    C = osipi.R1_to_C_linear_relaxivity(R1_list, R10, r1)
    np.testing.assert_allclose(C_truth, C, rtol=0, atol=1e-7)

    # Python tuple input
    R1_tuple = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    C = osipi.R1_to_C_linear_relaxivity(R1_tuple, R10, r1)
    np.testing.assert_allclose(C_truth, C, rtol=0, atol=1e-7)

    # Test with float32 dtype
    R1_32 = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    R10_32 = np.float32(1)
    r1_32 = np.float32(5)
    C = osipi.R1_to_C_linear_relaxivity(R1_32, R10_32, r1_32)
    np.testing.assert_allclose(C_truth, C, rtol=1e-6, atol=1e-6)

    # Test with invalid input types
    with pytest.raises(TypeError):
        osipi.R1_to_C_linear_relaxivity(42, R10, r1)  # Not array-like

    with pytest.raises(TypeError):
        osipi.R1_to_C_linear_relaxivity(np.array([[1, 2], [3, 4]]), R10, r1)  # Not 1D


def test_aif_parker_array_like_inputs():
    # Test with different array-like input types

    # NumPy array input
    t_array = np.arange(0, 60, 1, dtype=np.float64)
    ca_array = osipi.aif_parker(t_array)

    # Python list input
    t_list = list(range(60))
    ca_list = osipi.aif_parker(t_list)
    np.testing.assert_allclose(ca_array, ca_list, rtol=1e-7, atol=1e-7)

    # Python tuple input
    t_tuple = tuple(range(60))
    ca_tuple = osipi.aif_parker(t_tuple)
    np.testing.assert_allclose(ca_array, ca_tuple, rtol=1e-7, atol=1e-7)

    # Test with float32 dtype
    t_32 = np.arange(0, 60, 1, dtype=np.float32)
    ca_32 = osipi.aif_parker(t_32)
    np.testing.assert_allclose(ca_array, ca_32, rtol=1e-6, atol=1e-6)

    # Test with invalid input types
    with pytest.raises(ValueError):
        osipi.aif_parker("not an array")


def test_tofts_array_like_inputs():
    # Test with different array-like input types

    # NumPy array inputs
    t_array = np.arange(0, 60, 1, dtype=np.float64)
    ca_array = osipi.aif_parker(t_array)
    Ktrans = np.float64(0.6)
    ve = np.float64(0.2)
    ct_array = osipi.tofts(t_array, ca_array, Ktrans, ve)

    # Python list inputs
    t_list = list(range(60))
    ca_list = list(osipi.aif_parker(t_list))
    ct_list = osipi.tofts(t_list, ca_list, Ktrans, ve)
    np.testing.assert_allclose(ct_array, ct_list, rtol=1e-7, atol=1e-7)

    # Python tuple inputs
    t_tuple = tuple(range(60))
    ca_tuple = tuple(osipi.aif_parker(t_tuple))
    ct_tuple = osipi.tofts(t_tuple, ca_tuple, Ktrans, ve)
    np.testing.assert_allclose(ct_array, ct_tuple, rtol=1e-7, atol=1e-7)

    # Mixed inputs - numpy array and list
    ct_mixed = osipi.tofts(t_array, ca_list, Ktrans, ve)
    np.testing.assert_allclose(ct_array, ct_mixed, rtol=1e-7, atol=1e-7)

    # Float32 dtype
    t_32 = np.arange(0, 60, 1, dtype=np.float32)
    ca_32 = osipi.aif_parker(t_32)
    Ktrans_32 = np.float32(0.6)
    ve_32 = np.float32(0.2)
    ct_32 = osipi.tofts(t_32, ca_32, Ktrans_32, ve_32)
    np.testing.assert_allclose(ct_array, ct_32, rtol=1e-5, atol=1e-5)

    # Test with invalid input types
    with pytest.raises(ValueError):
        osipi.tofts("invalid", ca_array, Ktrans, ve)
    with pytest.raises(ValueError):
        osipi.tofts(t_array, "invalid", Ktrans, ve)


def test_extended_tofts_array_like_inputs():
    # Test with different array-like input types

    # NumPy array inputs
    t_array = np.arange(0, 60, 1, dtype=np.float64)
    ca_array = osipi.aif_parker(t_array)
    Ktrans = np.float64(0.6)
    ve = np.float64(0.2)
    vp = np.float64(0.1)
    ct_array = osipi.extended_tofts(t_array, ca_array, Ktrans, ve, vp)

    # Python list inputs
    t_list = list(range(60))
    ca_list = list(osipi.aif_parker(t_list))
    ct_list = osipi.extended_tofts(t_list, ca_list, Ktrans, ve, vp)
    np.testing.assert_allclose(ct_array, ct_list, rtol=1e-7, atol=1e-7)

    # Python tuple inputs
    t_tuple = tuple(range(60))
    ca_tuple = tuple(osipi.aif_parker(t_tuple))
    ct_tuple = osipi.extended_tofts(t_tuple, ca_tuple, Ktrans, ve, vp)
    np.testing.assert_allclose(ct_array, ct_tuple, rtol=1e-7, atol=1e-7)

    # Mixed inputs - numpy array and list
    ct_mixed = osipi.extended_tofts(t_array, ca_list, Ktrans, ve, vp)
    np.testing.assert_allclose(ct_array, ct_mixed, rtol=1e-7, atol=1e-7)

    # Float32 dtype
    t_32 = np.arange(0, 60, 1, dtype=np.float32)
    ca_32 = osipi.aif_parker(t_32)
    Ktrans_32 = np.float32(0.6)
    ve_32 = np.float32(0.2)
    vp_32 = np.float32(0.1)
    ct_32 = osipi.extended_tofts(t_32, ca_32, Ktrans_32, ve_32, vp_32)
    np.testing.assert_allclose(ct_array, ct_32, rtol=1e-5, atol=1e-5)

    # Test with invalid input types
    with pytest.raises(ValueError):
        osipi.extended_tofts("invalid", ca_array, Ktrans, ve, vp)
    with pytest.raises(ValueError):
        osipi.extended_tofts(t_array, "invalid", Ktrans, ve, vp)


if __name__ == "__main__":
    test_signal_linear_array_like_inputs()
    test_signal_SPGR_array_like_inputs()
    test_S_to_R1_SPGR_array_like_inputs()
    test_R1_to_C_linear_relaxivity_array_like_inputs()
    test_aif_parker_array_like_inputs()
    test_tofts_array_like_inputs()
    test_extended_tofts_array_like_inputs()

    print("All array-like input tests passed!!")
