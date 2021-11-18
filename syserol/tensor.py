"""
Tensor decomposition methods
"""
import numpy as np
from scipy.optimize import minimize
from syserol.COVID import Tensor4D
import tensorly as tl
from scipy.optimize._numdiff import approx_derivative
from tensorly.cp_tensor import cp_lstsq_grad
from tensorly.tenalg import khatri_rao
from statsmodels.multivariate.pca import PCA
from copy import deepcopy
from .COVID import Tensor4D, dayLabels


tl.set_backend('numpy')


def calcR2X(tFac, tIn):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    vTop, vBottom = 0.0, 0.0

    tMask = np.isfinite(tIn)
    tFill = np.nan_to_num(tIn)
    vTop += np.sum(np.square(tl.cp_to_tensor(tFac) * tMask - tFill))
    vBottom += np.sum(np.square(tFill))

    return 1.0 - vTop / vBottom


def tensor_degFreedom(tFac) -> int:
    """ Calculate the degrees of freedom within a tensor factorization. """
    deg = np.sum([f.size for f in tFac.factors])

    return deg


def reorient_factors(tFac):
    """ This function ensures that factors are negative on at most one direction. """
    # Flip the subjects to be positive
    rMeans = np.sign(np.mean(tFac.factors[1], axis=0))
    agMeans = np.sign(np.mean(tFac.factors[2], axis=0))
    tFac.factors[0] *= rMeans[np.newaxis, :] * agMeans[np.newaxis, :]
    tFac.factors[1] *= rMeans[np.newaxis, :]
    tFac.factors[2] *= agMeans[np.newaxis, :]

    return tFac


def totalVar(tFac):
    """ Total variance of a factorization on reconstruction. """
    varr = tl.cp_norm(tFac)
    return varr


def sort_factors(tFac):
    """ Sort the components from the largest variance to the smallest. """
    rr = tFac.rank
    tensor = deepcopy(tFac)
    vars = np.array([totalVar(delete_component(tFac, np.delete(np.arange(rr), i))) for i in np.arange(rr)])
    order = np.flip(np.argsort(vars))

    tensor.weights = tensor.weights[order]
    tensor.factors = [fac[:, order] for fac in tensor.factors]
    np.testing.assert_allclose(tl.cp_to_tensor(tFac), tl.cp_to_tensor(tensor))

    return tensor


def delete_component(tFac, compNum):
    """ Delete the indicated component. """
    tensor = deepcopy(tFac)
    compNum = np.array(compNum, dtype=int)

    # Assert that component # don't exceed range, and are unique
    assert np.amax(compNum) < tensor.rank
    assert np.unique(compNum).size == compNum.size

    tensor.rank -= compNum.size
    tensor.weights = np.delete(tensor.weights, compNum)

    if hasattr(tFac, 'mFactor'):
        tensor.mFactor = np.delete(tensor.mFactor, compNum, axis=1)

    tensor.factors = [np.delete(fac, compNum, axis=1) for fac in tensor.factors]
    return tensor


def censored_lstsq(A: np.ndarray, B: np.ndarray, uniqueInfo) -> np.ndarray:
    """Solves least squares problem subject to missing data.
    Note: uses a for loop over the missing patterns of B, leading to a
    slower but more numerically stable algorithm
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """
    X = np.empty((A.shape[1], B.shape[1]))
    # Missingness patterns
    unique, uIDX = uniqueInfo

    for i in range(unique.shape[1]):
        uI = uIDX == i
        uu = np.squeeze(unique[:, i])

        Bx = B[uu, :]
        X[:, uI] = np.linalg.lstsq(A[uu, :], Bx[:, uI], rcond=None)[0]
    return X.T


def sigmoid(x: np.ndarray, P: np.ndarray):
    """ Basic sigmoidal function """
    x0, k = P
    y = 1 / (1 + np.exp(-k*(x-x0)))
    return y


def build_factor(v, P, rank):
    """ Builds our continous dimension factor given a parameter matrix P"""
    P = np.reshape(P, (-1, rank))
    factor = np.empty((v.size, P.shape[1]), dtype=P.dtype)
    for comp in range(P.shape[1]):
        factor[:, comp] = sigmoid(v, P[:, comp])

    return factor


def continue_R2X(p_init, v, tFac, tFill, tMask):
    """ Calculates R2X with current guess for tFac,
    which uses our current parameter to solve for continuous factor.
    Returns a negative R2X for minimization. """
    tFac.factors[3] = build_factor(v, p_init, tFac.rank)
    grad, sse = cp_lstsq_grad(tFac, tFill, return_loss=True, mask=tMask)
    grad = grad.factors[3].flatten()
    J = approx_derivative(lambda x: build_factor(v, x, tFac.rank).flatten(), p_init, method="cs")
    outt = grad @ J # Apply chain rule
    return sse, outt


def continuous_maximize_R2X(tFac, tOrig, v, P):
    """ Maximizes R2X of tFac with respect to parameter matrix P,
    which will be used to calculate our continuous factor.
    Thus, solves for continous factor while maximizing R2X. 
    Returns current guesses for:
    Parameter matrix P_updt: params x r matrix
    Continous factor: r x n matrix 
    """
    tMask = np.isfinite(tOrig)
    tFill = np.nan_to_num(tOrig)
    res = minimize(continue_R2X, P.flatten(), jac=True, args=(v, tFac, tFill, tMask))
    P_updt = np.reshape(res.x, (-1, tFac.rank))
    return P_updt, build_factor(v, P_updt, tFac.rank)


def cp_normalize(tFac):
    """ Normalize the factors using the inf norm. """
    for i, factor in enumerate(tFac.factors):
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        tFac.weights *= scales
        if i == 0 and hasattr(tFac, 'mFactor'):
            tFac.mFactor *= scales

        tFac.factors[i] /= scales

    return tFac


def initialize_cp(tensor: np.ndarray, rank: int):
    r"""Initialize factors used in `parafac`.
    Parameters
    ----------
    tensor : ndarray
    rank : int
    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    factors = []
    factors.append(np.ones((tensor.shape[0],rank)))
    # first and last mode have to be initialized to ones, cannot be solved with PCA due to missingness structure
    for mode in range(1, tl.ndim(tensor) - 1):
        unfold = tl.unfold(tensor, mode)

        # Remove completely missing columns
        unfold = unfold[:, np.sum(np.isfinite(unfold), axis=0) > 2]

        # Impute by PCA
        outt = PCA(unfold, ncomp=1, method="nipals", missing="fill-em", standardize=False, demean=False, normalize=False, max_em_iter=1000)
        recon_pca = outt.scores @ outt.loadings.T
        unfold[np.isnan(unfold)] = recon_pca[np.isnan(unfold)]

        U = np.linalg.svd(unfold)[0]

        if U.shape[1] < rank:
            # This is a hack but it seems to do the job for now
            pad_part = np.random.rand(U.shape[0], rank - U.shape[1])
            U = tl.concatenate([U, pad_part], axis=1)

        factors.append(U[:, :rank])
    # append last mode factors
    factors.append(np.ones((tensor.shape[3],rank)))

    return tl.cp_tensor.CPTensor((None, factors))


def perform_CMTF(tOrig=None, r=6):
    """ Perform CMTF decomposition. """
    if tOrig is None:
        tOrig, _ = Tensor4D()

    tFac = initialize_cp(tOrig, r)

    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]

    R2X_last = -np.inf
    tFac.R2X = calcR2X(tFac, tOrig)

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    # get unique days into vector format for continuous solve
    days = dayLabels()
    # initialize parameter matrix
    P = np.ones((2, r))

    for ii in range(200):
        print(ii)
        # PARAFAC on all modes
        for m in range(0, len(tFac.factors) - 1):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])

        # Solve for P and continuous factor by maximizing R2X
        P, tFac.factors[3] = continuous_maximize_R2X(tFac, tOrig, days, P)

        R2X_last = tFac.R2X
        tFac.R2X = calcR2X(tFac, tOrig)
        assert tFac.R2X > 0.0
        print(tFac.R2X)

        if tFac.R2X - R2X_last < 1e-4:
            break

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)

    if r > 1:
        tFac = sort_factors(tFac)

    print(tFac.R2X)

    return tFac
