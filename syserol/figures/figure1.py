from operator import sub
import numpy as np
from .common import getSetup, subplotLabel
from syserol.tensor3D import Tensor3D, reorient_factors_3D, cp_normalize_3D, sort_factors_3D
from syserol.COVID import Tensor4D
from syserol.tensor import calcR2X, perform_CMTF, tensor_degFreedom
from matplotlib.ticker import ScalarFormatter
from tensorly.decomposition import parafac
from tensorpack import perform_CP


def makeFigure():
    ax, f = getSetup((7, 3), (1, 2))
    comps = np.arange(1, 8)

    tFacR2X = np.zeros(comps.shape)
    sizeTfac = np.zeros(comps.shape)
    tensor, _ = Tensor4D()
    
    for i, cc in enumerate(comps):
        # Run factorization with continuous solve
        tFac = perform_CMTF(tensor, cc)
        tFacR2X[i] = tFac.R2X
        sizeTfac[i] = tensor_degFreedom(tFac)

    # Run factorization for 3D tensor
    tensor_3D, _ = Tensor3D()
    CPfacs = [parafac(tensor_3D, cc, tol=1e-10, n_iter_max=1000,
                        linesearch=True, orthogonalise=2) for cc in comps]
    sizeCP = [tensor_degFreedom(f, continuous=False) for f in CPfacs]
    # Normalize 3D factors
    CPfacs = [cp_normalize_3D(f) for f in CPfacs]
    CPfacs = [reorient_factors_3D(f) for f in CPfacs]
    CPfacs = [sort_factors_3D(f) if i > 0 else f for i,
                f in enumerate(CPfacs)]
    # Calculate R2X
    R2X_3D = np.array([calcR2X(f, tensor_3D, continuous=False) for f in CPfacs])


    ax[0].scatter(comps, tFacR2X, s=10)
    ax[0].set_ylabel("CMTF R2X")
    ax[0].set_xlabel("Number of Components")
    ax[0].set_xticks([x for x in comps])
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0.5, np.amax(comps) + 0.5)

    ax[1].set_xscale("log", base=2)
    ax[1].plot(sizeTfac, 1.0 - tFacR2X, ".", label="4D Continuous Tensor Factorization")
    ax[1].plot(sizeCP, 1.0 - R2X_3D, ".", label="3D Tensor Factorization")
    ax[1].set_ylabel("Normalized Unexplained Variance")
    ax[1].set_xlabel("Size of Reduced Data")
    ax[1].set_ylim(bottom=0.0)
    ax[1].set_xlim(2 ** 8 - 100, 2 ** 11 + 100)
    ax[1].xaxis.set_major_formatter(ScalarFormatter())
    ax[1].legend()

    subplotLabel(ax)
    return f