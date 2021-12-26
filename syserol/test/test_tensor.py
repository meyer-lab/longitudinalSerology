"""
Unit test file.
"""
import numpy as np
import tensorly as tl
from tensorly.cp_tensor import _validate_cp_tensor
from tensorly.random import random_cp
from ..tensor import perform_CMTF, delete_component, calcR2X, sort_factors
from ..COVID import Tensor4D


def test_R2X():
    """ Test to ensure R2X for higher components is larger. """
    arr = []
    for i in range(1, 7):
        facT = perform_CMTF(r=i)
        assert np.all(np.isfinite(facT.factors[0]))
        assert np.all(np.isfinite(facT.factors[1]))
        assert np.all(np.isfinite(facT.factors[2]))
        arr.append(facT.R2X)
    print("R2X:", facT.R2X)
    for j in range(len(arr) - 1):
        assert arr[j] < arr[j + 1]
    # confirm R2X is >= 0 and <=1
    assert np.min(arr) >= 0
    assert np.max(arr) <= 1


def test_delete():
    """ Test deleting a component results in a valid tensor. """
    tOrig, _ = Tensor4D()
    facT = perform_CMTF(tOrig, r=4)

    fullR2X = calcR2X(facT, tOrig)

    for ii in range(facT.rank):
        facTdel = delete_component(facT, ii)
        _validate_cp_tensor(facTdel)

        delR2X = calcR2X(facTdel, tOrig)

        assert delR2X < fullR2X


def test_sort():
    """ Test that sorting does not affect anything. """
    tOrig, _ = Tensor4D()
    tFac = random_cp(tOrig.shape, 8)
    tFac.cFactor = np.ones((4, 8))

    R2X = calcR2X(tFac, tOrig)
    tRec = tl.cp_to_tensor(tFac)

    tFac = sort_factors(tFac)
    sR2X = calcR2X(tFac, tOrig)
    stRec = tl.cp_to_tensor(tFac)

    np.testing.assert_allclose(R2X, sR2X)
    np.testing.assert_allclose(tRec, stRec)
