"""
Tensor decomposition methods
"""
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize, Bounds
from syserol.COVID import Tensor4D
import tensorly as tl
from scipy.optimize._numdiff import approx_derivative
from tensorly.cp_tensor import cp_lstsq_grad
from tensorly.tenalg import khatri_rao
from tensorpack import initialize_cp
from copy import deepcopy
from .COVID import Tensor4D, dayLabels


tl.set_backend('numpy')


def calcR2X(tFac, tIn):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    np.testing.assert_allclose(build_cFactor(tFac, tFac.cFactor), tFac.factors[3])
    vTop, vBottom = 0.0, 0.0

    tMask = np.isfinite(tIn)
    tFill = np.nan_to_num(tIn)
    vTop += np.sum(np.square(tl.cp_to_tensor(tFac) * tMask - tFill))
    vBottom += np.sum(np.square(tFill))

    return 1.0 - vTop / vBottom


def tensor_degFreedom(tFac) -> int:
    """ Calculate the degrees of freedom within a tensor factorization. """
    deg = np.sum([f.size for f in tFac.factors[0:3]]) + tFac.cFactor.size

    return deg


def reorient_factors(tFac):
    """ This function ensures that factors are negative on at most one direction. """
    # Flip the subjects to be positive
    rMeans = np.sign(np.mean(tFac.factors[1], axis=0))
    agMeans = np.sign(np.mean(tFac.factors[2], axis=0))
    tMeans = np.sign(np.mean(tFac.factors[3], axis=0))
    tFac.factors[0] *= (rMeans * agMeans * tMeans)[np.newaxis, :]
    tFac.factors[1] *= rMeans[np.newaxis, :]
    tFac.factors[2] *= agMeans[np.newaxis, :]
    tFac.factors[3] *= tMeans[np.newaxis, :]

    tFac.cFactor[0:2, :] /= tMeans[np.newaxis, :]
    np.testing.assert_allclose(build_cFactor(tFac, tFac.cFactor), tFac.factors[3])
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

    np.testing.assert_allclose(build_cFactor(tFac, tFac.cFactor), tFac.factors[3])
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

    tensor.factors = [np.delete(fac, compNum, axis=1) for fac in tensor.factors]
    tensor.cFactor = np.delete(tensor.cFactor, compNum, axis=1)

    np.testing.assert_allclose(build_cFactor(tFac, tFac.cFactor), tFac.factors[3])
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


def curve(x: np.ndarray, P: np.ndarray):
    """ Function
    y(t) = d + (a − d)/(1 + (t/c)b) 
    Based on Zohar et al. curve
    P will be a 4 element array now.
    """
    a, b, c, d = P
    y = b + ((a - b) / (1.0 + np.power(x / c, d)))
    assert np.all(np.isfinite(y))
    return y


def build_cFactor(tFac, P):
    """ Builds our continous dimension factor given a parameter matrix P"""
    P = np.reshape(P, (-1, tFac.rank))
    factor = np.empty((tFac.time.size, P.shape[1]), dtype=P.dtype)
    for comp in range(P.shape[1]):
        factor[:, comp] = curve(tFac.time, P[:, comp])

    return factor


def continue_R2X(p_init, tFac, tFill, tMask):
    """ Calculates R2X with current guess for tFac,
    which uses our current parameter to solve for continuous factor.
    Returns a negative R2X for minimization. """
    tFac.factors[3] = build_cFactor(tFac, p_init)
    grad, sse = cp_lstsq_grad(tFac, tFill, return_loss=True, mask=tMask)
    grad = grad.factors[3].flatten()
    J = approx_derivative(lambda x: build_cFactor(tFac, x).flatten(), p_init, method="3-point")
    return sse, grad @ J # Apply chain rule


def continuous_maximize_R2X(tFac, tOrig):
    """ Maximizes R2X of tFac with respect to parameter matrix P,
    which will be used to calculate our continuous factor.
    Thus, solves for continous factor while maximizing R2X. 
    Returns current guesses for:
    Parameter matrix P_updt: params x r matrix
    Continous factor: r x n matrix 
    """
    tMask = np.isfinite(tOrig)
    tFill = np.nan_to_num(tOrig)

    # Setup bounds
    lb = tFac.cFactor.copy()
    ub = tFac.cFactor.copy()
    lb[0:2, :] = -np.inf
    ub[0:2, :] = np.inf
    lb[2, :] = 0.1
    ub[2, :] = np.inf
    lb[3, :] = 0.1
    ub[3, :] = 10.0
    bnds = Bounds(lb.flatten(), ub.flatten(), keep_feasible=True)

    res = minimize(continue_R2X, tFac.cFactor.flatten(), jac=True, bounds=bnds, args=(tFac, tFill, tMask))
    P_updt = np.reshape(res.x, (-1, tFac.rank))
    return P_updt, build_cFactor(tFac, P_updt)


def cp_normalize(tFac):
    """ Normalize the factors using the inf norm. """
    for i in range(len(tFac.factors)):
        scales = np.linalg.norm(tFac.factors[i], ord=np.inf, axis=0)
        tFac.weights *= scales
        tFac.factors[i] /= scales

        if i == 3:
            tFac.cFactor[0:2, :] /= scales[np.newaxis, :]

    np.testing.assert_allclose(build_cFactor(tFac, tFac.cFactor), tFac.factors[3])
    return tFac


def check_unimodality(arr):
    arrDiff = np.diff(arr, axis=0)
    diffMin = np.min(arrDiff, axis=0)
    diffMax = np.max(arrDiff, axis=0)
    assert np.all(diffMin * diffMax >= 0.0)


def perform_CMTF(tOrig=None, r=6, tol=1e-4, maxiter=300):
    """ Perform CMTF decomposition. """
    if tOrig is None:
        tOrig, _ = Tensor4D()

    tFac = initialize_cp(tOrig, r)

    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]

    tFac.R2X = -np.inf

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    # get unique days into vector format for continuous solve
    tFac.time = dayLabels()
    # initialize parameter matrix
    # with Zohar curve, P has 4 parameters
    tFac.cFactor = np.ones((4, r))

    tq = tqdm(range(maxiter))
    for _ in tq:
        # PARAFAC on all modes
        for m in range(0, len(tFac.factors) - 1):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])

        # Solve for P and continuous factor by maximizing R2X
        tFac.cFactor, tFac.factors[3] = continuous_maximize_R2X(tFac, tOrig)

        # assert that every continuous factor is unimodal (increases or decreases in one direction)
        check_unimodality(tFac.factors[3])

        R2X_last = tFac.R2X
        tFac.R2X = calcR2X(tFac, tOrig)
        assert tFac.R2X > 0.0
        tq.set_postfix(R2X=tFac.R2X, refresh=False)

        if tFac.R2X - R2X_last < tol:
            break

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)

    if r > 1:
        tFac = sort_factors(tFac)

    return tFac
