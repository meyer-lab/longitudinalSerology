"""
Unit test file.
"""
from ..COVID import pbsSubtractOriginal, Tensor4D
from ..kaplonek import kaplonek_4D


def test_COVID_import():
    """ Test COVID import functions. """
    pbsSubtractOriginal()
    tensor, _ = Tensor4D()
    tt, axes = kaplonek_4D()
    for ii in range(len(axes)):
        assert tt.shape[ii] == len(axes[ii])
