""" Generate a simulated tensor for testing methods """

from sklearn import impute
import tensorly as tl
import seaborn as sns
import pandas as pd
import numpy as np
from syserol.figures.common import getSetup
from syserol.figures.figure1 import makeFigure
from syserol.tensor import build_cFactor, continue_R2X, perform_CMTF, curve
from syserol.COVID import Tensor4D, dayLabels
from tensorpack.decomposition import Decomposition, entry_drop


def generate_simulated(r=6):
    tensor, _ = Tensor4D()
    # Generate random CP tensor, in factor form with our COVID tensor shape
    random_cp = tl.random.random_cp(shape=tensor.shape, rank=r)

    # initialize time factor and continuous factor
    random_cp.time = dayLabels()
    # Generate P matrix with realistic values.
    # Based on curve simulation, a, b, c, and d can/will be in the ranges below
    P = np.zeros((4, r))
    for i in range(r):
        a = np.random.uniform(0.01, 1)
        b = np.random.uniform(0.01, 1.2)
        c = np.random.uniform(0.01, 30)
        d = np.random.uniform(0.01, 100)
        P[:, i] = a, b, c, d

    timefac = build_cFactor(random_cp, P)
    # assign random continuous factor to our simulated tensor
    random_cp.factors[3] = timefac
    # build the tensor from factors (3 regular CP factors, 1 continuous factor)
    sim_tensor = tl.cp_to_tensor(random_cp)

    return sim_tensor, P


def imputeSim(sim_tensor, missingness):
    entry_drop(sim_tensor, int(np.sum(np.isfinite(sim_tensor))*missingness)) # modify tensor in place
    

def viewP(P):
    time = dayLabels()
    arr = np.stack((curve(time, P[:,i]) for i in range(P.shape[1])), axis=1)
    df = pd.DataFrame(arr, columns=["Component " + str(ii + 1) for ii in range(P.shape[1])])
    df["Time (days)"] = time
    df = df.set_index("Time (days)")

    ax, f = getSetup((7,5), (1,1))
    sns.lineplot(data=df, ax=ax[0])
    return f
