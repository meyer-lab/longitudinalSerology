"""
Unit test file.
"""
import numpy as np
from ..COVID import pbsSubtractOriginal, Tensor4D


def test_COVID_import():
    """ Test COVID import functions. """
    pbsSubtractOriginal()
    tensor, _ = Tensor4D()
