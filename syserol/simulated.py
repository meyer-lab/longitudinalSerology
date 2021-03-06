""" Generate a simulated tensor for testing methods """

from random import random
from sklearn import impute
import tensorly as tl
import seaborn as sns
import pandas as pd
import numpy as np
from syserol.figures.common import getSetup
from syserol.figures.figure1 import makeFigure
from syserol.tensor import build_cFactor, reorient_factors, sort_factors, cp_normalize, curve
from syserol.COVID import Tensor4D, dayLabels
from tensorpack.decomposition import Decomposition, entry_drop


def generate_simulated(r=4, rand=False):
    # Generate random CP tensor, in factor form with our COVID tensor shape
    random_cp = tl.random.random_cp(shape=(226, 6, 11, 38), rank=r)

    # initialize time factor and continuous factor
    random_cp.time = dayLabels()
    # Generate P matrix with realistic values.
    # Based on curve simulation, a, b, c, and d can/will be in the ranges below
    if rand:
        P = np.zeros((4, r))
        for i in range(r):
            a = np.random.uniform(0.01, 1)
            b = np.random.uniform(0.01, 1.2)
            c = np.random.uniform(0.01, 30)
            d = np.random.uniform(0.01, 100)
            P[:, i] = a, b, c, d
    else:
        assert r == 4
        P = np.array([[0.05, 0.1, 0.2, 0.7], [1, 1.1, 1, .95], [20, 2, 30, 8], [30, 2, 5, 10]])

    timefac = build_cFactor(random_cp, P)
    # assign random continuous factor to our simulated tensor
    random_cp.factors[3] = timefac
    random_cp.cFactor = P

    # run reorient, normalize, and sort beforehand
    random_cp = cp_normalize(random_cp)
    random_cp = reorient_factors(random_cp)
    random_cp = sort_factors(random_cp)

    # build the tensor from factors (3 regular CP factors, 1 continuous factor)
    sim_tensor = tl.cp_to_tensor(random_cp)

    return sim_tensor, P, random_cp


def imputeSim(sim_tensor, missingness):
    entry_drop(sim_tensor, int(np.sum(np.isfinite(sim_tensor))*missingness)) # modify tensor in place
    

def viewP(P):
    """ Returns a dataframe of the current curves generated by P for plotting"""
    time = dayLabels()
    arr = np.stack((curve(time, P[:,i]) for i in range(P.shape[1])), axis=1)
    # enforce same bounds that our decomp enforces
    tMeans = np.sign(P[1,:] - P[0,:])
    arr *= tMeans[np.newaxis, :]
    df = pd.DataFrame(arr, columns=["Component " + str(ii + 1) for ii in range(P.shape[1])])
    df["Time (days)"] = time
    df = df.set_index("Time (days)")
    
    return df
