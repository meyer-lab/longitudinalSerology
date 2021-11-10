"""
Tensor decomposition methods
"""
import numpy as np
import scipy as sp
from syserol.COVID import Tensor4D
import tensorly as tl
from tensorly.tenalg import khatri_rao
from statsmodels.multivariate.pca import PCA
from copy import deepcopy
from .COVID import Tensor4D, dayLabels


tl.set_backend('numpy')


def buildGlycan(tFac):
    """ Build the glycan matrix from the factors. """
    return tFac.factors[0] @ tFac.mFactor.T


def calcR2X(tFac, tIn=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    assert (tIn is not None)

    vTop, vBottom = 0.0, 0.0

    if tIn is not None:
        tMask = np.isfinite(tIn)
        vTop += np.sum(np.square(tl.cp_to_tensor(tFac) * tMask - np.nan_to_num(tIn)))
        vBottom += np.sum(np.square(np.nan_to_num(tIn)))

    return 1.0 - vTop / vBottom


def tensor_degFreedom(tFac) -> int:
    """ Calculate the degrees of freedom within a tensor factorization. """
    deg = np.sum([f.size for f in tFac.factors])

    if hasattr(tFac, 'mFactor'):
        deg += tFac.mFactor.size

    return deg


def reorient_factors(tFac):
    """ This function ensures that factors are negative on at most one direction. """
    # Flip the subjects to be positive
    rMeans = np.sign(np.mean(tFac.factors[1], axis=0))
    agMeans = np.sign(np.mean(tFac.factors[2], axis=0))
    tFac.factors[0] *= rMeans[np.newaxis, :] * agMeans[np.newaxis, :]
    tFac.factors[1] *= rMeans[np.newaxis, :]
    tFac.factors[2] *= agMeans[np.newaxis, :]

    if hasattr(tFac, 'mFactor'):
        tFac.mFactor *= rMeans[np.newaxis, :] * agMeans[np.newaxis, :]

    return tFac


def totalVar(tFac):
    """ Total variance of a factorization on reconstruction. """
    varr = tl.cp_norm(tFac)
    if hasattr(tFac, 'mFactor'):
        varr += tl.cp_norm((None, [tFac.factors[0], tFac.mFactor]))
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

    if hasattr(tFac, 'mFactor'):
        tensor.mFactor = tensor.mFactor[:, order]
        np.testing.assert_allclose(buildGlycan(tFac), buildGlycan(tensor))

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


def F(P: np.ndarray, x: np.ndarray, y: np.ndarray):
    return(sigmoid(x, P) - y)


def reverse_p(x: np.ndarray, y: np.ndarray, p_init: np.ndarray):
    """ Solves for parameter matrix given an output y from our sigmoidal time function 
    In this case, x is our time vector, y is our solution to the sigmoidal function.
    """
    res = sp.optimize.least_squares(F, p_init, args=(x, y))
    return(res.x)


def continue_solve(v: np.ndarray, A_hat: np.ndarray):
    """ Reverse solves for parameter matrix P_new given current guess for time factor.
    Updates current iterate of time factor (A_new) using solved parameters. 
    P_new : params x r matrix
    A_new : r x n matrix 
    Note that the first iteration initializes P to be all ones.
    """
    rank = A_hat.shape[1]
    A_new = np.empty(A_hat.shape)
    P_new = np.empty((4, rank))

    for comp in range(rank):
        p_est = np.polyfit(v, A_hat[:, comp], 3)
        P_new[:, comp] = p_est
        A_new[:, comp] = np.polyval(p_est, v)

    return A_new


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

    for ii in range(2000):
        # PARAFAC on all modes
        for m in range(0, len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])
        
        # for final (continuous) dimension, solve for P and recalculate continuous factor
        tFac.factors[m] = continue_solve(days, tFac.factors[m])

        if ii % 2 == 0:
            R2X_last = tFac.R2X
            tFac.R2X = calcR2X(tFac, tOrig)
            assert tFac.R2X > 0.0

        if tFac.R2X - R2X_last < 1e-6:
            break

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)

    if r > 1:
        tFac = sort_factors(tFac)

    print(tFac.R2X)

    return tFac
