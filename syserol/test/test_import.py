"""
Unit test file.
"""
from ..COVID import pbsSubtractOriginal, Tensor4D
from ..kaplonek import kaplonek_4D


def test_COVID_import():
    """ Test COVID import functions. """
    pbsSubtractOriginal()
    tensor, _ = Tensor4D()
    tt, _ = kaplonek_4D()
