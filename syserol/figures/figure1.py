from operator import sub
import numpy as np
from .common import getSetup, subplotLabel
from syserol.COVID import Tensor4D
from syserol.tensor import calcR2X, perform_CMTF, tensor_degFreedom
from matplotlib.ticker import ScalarFormatter
from tensorly.decomposition import parafac
from tensorpack import perform_CP


def makeFigure():
    ax, f = getSetup((13, 9), (1, 1))
    comps = np.arange(1, 8)

    tFacR2X = np.zeros(comps.shape)
    sizeTfac = np.zeros(comps.shape)
    CPR2X = np.zeros(comps.shape)
    sizeCP = np.zeros(comps.shape)

    tensor, _ = Tensor4D()
    
    for i, cc in enumerate(comps):
        # Run factorization with continuous solve
        tFac = perform_CMTF(tensor, cc)
        tFacR2X[i] = tFac.R2X
        sizeTfac[i] = tensor_degFreedom(tFac)

        # Run factorization with standard CP
        CP = perform_CP(tensor, cc)
        CPR2X[i] = CP.R2X
        sizeCP[i] = tensor_degFreedom(CP, continuous=False)
    
    ax[0].set_xscale("log", base=2)
    ax[0].plot(sizeTfac, 1.0 - tFacR2X, ".", label="Continuous Factorization")
    ax[0].plot(sizeCP, 1.0 - CPR2X, ".", label="CP Factorization")
    ax[0].set_ylabel("Normalized Unexplained Variance")
    ax[0].set_xlabel("Size of Reduced Data")
    ax[0].set_ylim(bottom=0.0)
    ax[0].set_xlim(2 ** 7, 2 ** 12)
    ax[0].xaxis.set_major_formatter(ScalarFormatter())
    ax[0].legend()

    subplotLabel(ax)
    return f