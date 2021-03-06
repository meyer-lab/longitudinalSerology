from operator import sub
import numpy as np
from .common import getSetup, subplotLabel
from syserol.tensor3D import Tensor3D, reorient_factors_3D, cp_normalize_3D, sort_factors_3D
from syserol.COVID import Tensor4D, earlyDaysdf
from syserol.tensor import calcR2X, flatten_to3D, perform_contTF, tensor_degFreedom
from matplotlib.ticker import ScalarFormatter
from tensorly.decomposition import parafac
from tensorpack import perform_CP


def makeFigure():
    """ Generate Figure 1, which runs generic R2X Plots function:
        Get R2X values for various methods on out COVID dataset """
    return R2X_Plots()


def R2X_Plots(tensor=None, tensor_3D=None, fig4=False):
    """ Generalized code for making R2X plots, capable of handling COVID or simulated data """
    ax, f = getSetup((8, 3), (1, 2))
    if fig4:
        comps = np.arange(1, 5)
    else:
        comps = np.arange(1, 8)

    tFacR2X = np.zeros(comps.shape)
    sizeTfac = np.zeros(comps.shape)
    CPR2X = np.zeros(comps.shape)
    sizeCP = np.zeros(comps.shape)

    if tensor is None:
        #df = earlyDaysdf()
        tensor, _ = Tensor4D()
    
    for i, cc in enumerate(comps):
        # Run factorization with continuous solve
        tFac = perform_contTF(tensor, cc)
        tFacR2X[i] = tFac.R2X
        sizeTfac[i] = tensor_degFreedom(tFac)

        # Run factorization with standard CP
        CP = perform_CP(tensor, cc)
        CPR2X[i] = CP.R2X
        sizeCP[i] = tensor_degFreedom(CP, continuous=False)

    # Run factorization for 3D tensor
    if tensor_3D is None:
        #tensor_3D = flatten_to3D(tensor)
        tensor_3D, _ = Tensor3D()
        CPfacs = [parafac(tensor_3D, cc, tol=1e-10, n_iter_max=1000,
                        linesearch=True, orthogonalise=2) for cc in comps]
    else:
         CPfacs = [perform_CP(tensor_3D, cc) for cc in comps]

    size3D = [tensor_degFreedom(f, continuous=False) for f in CPfacs]
    # Normalize 3D factors
    CPfacs = [cp_normalize_3D(f) for f in CPfacs]
    CPfacs = [reorient_factors_3D(f) for f in CPfacs]
    CPfacs = [sort_factors_3D(f) if i > 0 else f for i,
                f in enumerate(CPfacs)]
    # Calculate R2X
    R2X_3D = np.array([calcR2X(f, tensor_3D, continuous=False) for f in CPfacs])

    print("Longitudinal R2X: ", tFacR2X, '\n')
    print("CP 4D R2X: ", CPR2X, '\n')
    print("3D CP R2X: ", R2X_3D, '\n')
    print("Size Longitudinal: ", sizeTfac, '\n')
    print("Size CP 4D: ", sizeCP, '\n')
    print("Size CP 3D: ", size3D, '\n')

    ax[0].scatter(comps, tFacR2X, s=16)
    ax[0].set_ylabel("Continuous Factorization R2X", fontsize=11.5)
    ax[0].set_xlabel("Number of Components", fontsize='x-large')
    ax[0].set_xticks([x for x in comps])
    ax[0].tick_params(axis='both', which='major', labelsize=10)
    ax[0].set_xticklabels([x for x in comps])
    ax[0].set_ylim(top=1.0)
    ax[0].set_xlim(0.5, np.amax(comps) + 0.5)

    ax[1].set_xscale("log", base=2)
    ax[1].plot(sizeTfac, 1.0 - tFacR2X, ".", markersize=8, label="4D Continuous Tensor Factorization")
    ax[1].plot(sizeCP, 1.0 - CPR2X, ".", markersize=8, label="4D CP Tensor Factorization")
    ax[1].plot(size3D, 1.0 - R2X_3D, ".", markersize=8, label="3D CP Tensor Factorization")
    ax[1].set_ylabel("Normalized Unexplained Variance", fontsize=11.5)
    ax[1].set_xlabel("Size of Reduced Data", fontsize='x-large')
    ax[1].tick_params(axis='both', which='major', labelsize=10)
    ax[1].set_ylim(bottom=0.0)
    if fig4:
        topval = 1.0 - tFacR2X[0]
        ax[1].set_ylim(top=topval + 0.02)
    #if fig4 is False:
    #    ax[1].set_xlim(2 ** 8 - 100, 2 ** 11 + 100)
    ax[1].xaxis.set_major_formatter(ScalarFormatter())
    ax[1].legend()

    subplotLabel(ax)
    return f
