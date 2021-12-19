from operator import sub
import numpy as np
from .common import getSetup, subplotLabel
from syserol.COVID import Tensor4D
from syserol.tensor import perform_CMTF, tensor_degFreedom
from matplotlib.ticker import ScalarFormatter

def makeFigure():
    ax, f = getSetup((13, 9), (1, 1))
    comps = np.arange(1, 8)

    tFacR2X = np.zeros(comps.shape)
    sizeTfac = np.zeros(comps.shape)
    tensor, _ = Tensor4D()
    
    for i, cc in enumerate(comps):
        tFac = perform_CMTF(tensor, cc)
        tFacR2X[i] = tFac.R2X
        sizeTfac[i] = tensor_degFreedom(tFac)
    
    ax[0].set_xscale("log", base=2)
    ax[0].plot(sizeTfac, 1.0 - tFacR2X, ".", label="Continuous Factorization")
    ax[0].set_ylabel("Normalized Unexplained Variance")
    ax[0].set_xlabel("Size of Reduced Data")
    ax[0].set_ylim(bottom=0.0)
    ax[0].set_xlim(2 ** 8, 2 ** 12)
    ax[0].xaxis.set_major_formatter(ScalarFormatter())
    ax[0].legend()

    subplotLabel(ax)
    return f