"""Vector and sorted-array utilities used by Mixle estimators.

The helpers cover special functions, sorted merges, count aggregation, and
linear-algebra conveniences shared by legacy estimation and evaluation code.
"""

from collections.abc import Iterable, Sequence
from typing import SupportsIndex, overload

import numpy as np
import scipy.linalg
import scipy.special

from mixle.engines.arithmetic import *


@overload
def gammaln(x: np.ndarray) -> np.ndarray:
    """Return log-gamma values for an ndarray input."""
    ...


@overload
def gammaln(x: float) -> float:
    """Return the scalar log-gamma value for a float input."""
    ...


def gammaln(x: np.ndarray | float | int) -> np.ndarray | float:
    """Return logrithm of the gamma function.

    Returns np.log(.np.abs(Gamma(x)))

    Args:
        x (Union[np.ndarray, float, int])): Takes numeric value of np.ndarray of float/int.

    Returns:
        log(Gamma(x)) as float if x is a float/int, or np.ndarray[np.float] if x is a numpy array.

    """
    # Return a Python float for any scalar input (float/int/np.floating/np.integer), as the
    # docstring and overloads promise; previously a python-int or np.float64 leaked a 0-d ndarray.
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(scipy.special.gammaln(x))

    return np.asarray(scipy.special.gammaln(x))


def sorted_merge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Performs the merge-step of merge sort on sorted np.ndarray's a and b, returning sorted array.

    Args:
        a (ndarray): Sorted numpy array.
        b (ndarray): Sorted numpy array.

    Returns:
        Sorted numpy array containing merge sorted a and b. Array len = len(a)+len(b).

    """
    if len(a) < len(b):
        b, a = a, b
    c = np.empty(len(a) + len(b), dtype=a.dtype)
    b_indices = np.arange(len(b)) + np.searchsorted(a, b)
    a_indices = np.ones(len(c), dtype=bool)
    a_indices[b_indices] = False
    c[b_indices] = b
    c[a_indices] = a

    return c


def sorted_dict_merge_add(
    k_vec1: np.ndarray, c_vec1: np.ndarray, k_vec2: np.ndarray, c_vec2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Performs a merge on two sorted arrays of dictionary keys and the counts for their respective keys.

    Returns the merge sorted keys and corresponding counts.

    Args:
        k_vec1 (ndarray): Numpy array of sorted dictionary keys.
        c_vec1 (ndarray): Numpy array of counts for keys in vector k_vec1.
        k_vec2 (ndarray): Numpy array of sorted dictionary keys.
        c_vec2 (ndarray): Numpy array of counts for keys in vector k_vec2.

    Returns:
        Tuple of numpy arrays containing the merge sorted dictionary keys and corresponding counts.
    """
    if len(k_vec2) == 0:
        return k_vec1, c_vec1
    elif len(k_vec1) == 0:
        return k_vec2, c_vec2

    if len(k_vec1) < len(k_vec2):
        return sorted_dict_merge_add(k_vec2, c_vec2, k_vec1, c_vec1)

    _, idx1, idx2 = np.intersect1d(k_vec1, k_vec2, assume_unique=True, return_indices=True)

    rv_vals = k_vec1.copy()
    rv_cnts = c_vec1.copy()
    rv_cnts[idx1] += c_vec2[idx2]
    new_vals = np.delete(k_vec2, idx2)
    new_cnts = np.delete(c_vec2, idx2)
    new_idx = np.searchsorted(rv_vals, new_vals)
    rv_vals = np.insert(rv_vals, new_idx, new_vals)
    rv_cnts = np.insert(rv_cnts, new_idx, new_cnts)

    return rv_vals, rv_cnts


def make(x: np.ndarray | Sequence[int | float | str] | list[np.ndarray]) -> np.ndarray:
    """Convert the array x into a numpy array.

    Args:
        x (Union[np.ndarray, Sequence[Union[int, float, str]]): Array like object that can be converted to a numpy
        array. E.g. lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.

    Returns:
        Numpy array conversion of x.

    """
    return np.asarray(x)


def make_pdf(x: np.ndarray | Sequence[float] | list[np.ndarray]):
    """Takes log density values and normalizes on the log-scale, returning an ndarray that s.t. np.exp(rv).sum() == 1.0.

    Arg data type for x: Union[np.ndarray, Sequence[float], List[np.ndarray]]).
    Args:
        x (See above): Array like object with float data type that can be converted to a numpy array. E.g. lists, lists
        of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    Returns:
        Returns an ndarray that s.t. np.exp(rv).sum() == 1.0.
    """
    rv = np.array(x, dtype=np.float64)
    n = len(rv)
    rv_max = rv.max()

    if rv_max == -inf:
        rv = zeros(n) - log(n)
    else:
        rv_sum = np.log(np.sum(np.exp(rv - rv_max))) + rv_max
        rv -= rv_sum

    return rv


def zeros(n: int | Iterable | tuple[int]) -> np.ndarray:
    """Return numpy array of shape n, with default dtype=float64.

    Args:
        n (Union[int, Iterable, Tuple[int]]): Shape tuple of ints, Iterable, or int.

    Returns:
        Return numpy array of shape n, with default dtype=float64.

    """
    return np.zeros(n)


def mat_inv(x: list[list[float | int]] | list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Computes the inverse of a square matrix x.

    Arg x data type Union[List[List[Union[float, int]]],List[np.ndarray], np.ndarray]).
    Args:
        x (See above): List of List[float/int], List of np.ndarray, or 2-d np.ndarray of square matrix.

    Returns:
        Inverse of x as 2-d numpy array.

    """
    return np.linalg.inv(x)


def dot(x: np.ndarray | Iterable | int | float, y: np.ndarray | Iterable | int | float) -> np.ndarray | float:
    """Performs call to numpy.dot().

    Args:
        x: Numpy array, array-like, or scalar.
        y: Numpy array, array-like, or scalar.
    Returns:
        Returns float/int if x and y are both 1d vectors, returns 1d vector if x xor y is scalar, and matrix else.

    """
    return np.dot(x, y)


def outer(x: np.ndarray | Iterable | int | float, y: np.ndarray | Iterable | int | float) -> np.ndarray:
    """Compute the outer product of two vectors

    Args:
        x:  (M,) array_like
        y:  (N,) array_like

    Returns: (M, N) ndarray.

    """
    return np.outer(x, y)


def diag(x: np.ndarray) -> np.ndarray:
    """Extract a diagonal or construct a diagonal array.

    Note: If x is 2-D return np.ndarray with diagonal. If x is 1-D returns 2-d diagonal matrix with x on diagonal.

    See the more detailed documentation for ``numpy.diagonal`` if you use this
    function to extract a diagonal and wish to write to the resulting array;
    whether it returns a copy or a view depends on what version of numpy you
    are using.

    Args:
        x: 2-D array, or 1-D array.
    Returns:
        The extracted diagonal or constructed diagonal array.

    """
    return np.diag(x)


def reshape(x: np.ndarray, sz: SupportsIndex | Sequence[SupportsIndex]) -> np.ndarray:
    """Gives a new shape to an array without changing its data.

    Args:
        x (np.ndarray): Array to be reshaped.
        sz (Tuple[int,...]): Shape compatible with size of array x.

    Return:
        Reshaped array containing elements of x with shape = sz.

    """
    return np.reshape(x, sz)


def cholesky(x_mat: np.ndarray) -> tuple[np.ndarray, bool] | None:
    """Compute the Cholesky decomposition of a matrix, to use in cho_solve.

    Returns a matrix containing the Cholesky decomposition, x_mat = L L* or x_mat = U* U of a Hermitian positive-definite
    matrix x_mat. The return value can be directly used as the first parameter to cho_solve.

    Args:
        x_mat (np.ndarray): Square np.ndarray of matrix to be decomposed.
    Returns:
        Square np.ndarray matrix whose upper or lower triangle contains the Cholesky factor of x. If Cholesky
            factor cannot be found None is returned.
    """
    try:
        rv = scipy.linalg.cho_factor(x_mat)
    except np.linalg.LinAlgError:
        rv = None

    return rv


def cho_solve(a_mat: tuple[np.ndarray, bool], b: np.ndarray) -> np.ndarray:
    """Solve the linear equations a_mat x = b, given the Cholesky factorization of a_mat.

    Args:
        a_mat (Tuple[np.ndarray, bool]): Cholesky factorization of a, as given by cho_factor.
        b (np.ndarray): Right-hand side np.ndarray in a_mat*x = b.

    Returns:
        The solution to the system a_mat*x = b.

    """
    return scipy.linalg.cho_solve(a_mat, b)


def maximum(
    x: float | int | Iterable | np.ndarray,
    y: float | int | Iterable | np.ndarray,
    output: float | int | np.ndarray | None = None,
) -> float | int | np.ndarray:
    """Element-wise maximum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Args:
        x (array-like): Array-like holding values to be compared. If ``x.shape != y.shape``, they must be broadcastable
            to a common shape (which becomes the shape of the output).
        y (array-like): Array-like holding values to be compared. If ``x.shape != y.shape``, they must be broadcastable
            to a common shape (which becomes the shape of the output).
        output: Optional np.ndarray of float to output results to.

    Returns:
        ndarray or scalar. The maximum of x and y, element-wise. This is a scalar if both x and y are scalars.
    """
    return np.maximum(x, y, output=output)


def log_sum(x: np.ndarray) -> float:
    """Performs log(sum(exp(x)) on 1-d numpy array. E.g. for x_i = log(y_i), log(sum(exp(x)) = log(sum(y)).

    Args:
        x (ndarray): Numpy array on log-scale. E.g. x_i = log(y_i).
    Returns:
        Float value log(sum(exp(x)), or -np.inf if max(x) is -np.inf.
    """
    max_val = np.max(x)

    if max_val == -np.inf:
        return -np.inf
    else:
        rv = x - max_val
        np.exp(rv, out=rv)
        return np.log(rv.sum()) + max_val


def weighted_log_sum(x: np.ndarray, w: np.ndarray) -> float:
    """Computes numerically stable log-sum-of-exponentials with weights=exp(w), on the observation values y=exp(x),
    returning log(sum(exp(x)*exp(w))).

    Note: The weights are on the log-scale.

    Args:
        x (ndarray): Numpy array on log-scale. E.g. x_i = log(y_i).
        w (ndarray): Numpy array on of weights for y_i = exp(x_i) on the log-scale. E.g. w_i = log(weight_i).

    Returns:
        Float value log(sum(exp(x)*exp(w))), or -np.inf if any x or w are -np.inf. Inputs are log-densities
        and log-weights (<= 0); +inf terms are not supported (this is a hot EM path).

    """
    y = x + w
    y[np.bitwise_or(np.isinf(x), np.isinf(w))] = -np.inf

    return log_sum(y)


def log_posterior(x: np.ndarray) -> np.ndarray:
    """Computes posterior density for vector of log-likelihood evaluated at each parameter component.

    I.e. if,

    x = [log(p_mat(obs_i | theta_0)), log(p_mat(obs_i | theta_1)),..., log(p_mat(obs_i | theta_{n-1}))],

    then returned value is,

    [log(p_mat(theta_0| obs_i)),...,log(p_mat(theta_{n-1}|obs_i))],

    where,

    log(p_mat(theta_j| obs_i)) = log(p_mat(obs_i| theta_j)) - log(p_mat(obs_i)).

    Args:
        x (np.ndarray): Numpy array of log-density values for each component/parameter value
            x = [log(p_mat(obs_i | theta_0)), log(p_mat(obs_i | theta_1)),..., log(p_mat(obs_i | theta_{n-1}))].

    Returns:
        Numpy array of log-posterior for each component/parameter value
            [log(p_mat(theta_0| obs_i)),...,log(p_mat(theta_{n-1}|obs_i))]. Returns numpy array of [-log(len(x))]
            if nan or inf detected in x.
    """
    max_val = x.max()

    if isinf(max_val) or isnan(max_val):
        return zeros(len(x)) - log(len(x))

    mass = log(exp(x - max_val).sum()) + max_val
    return x - mass


def posterior(
    log_x: np.ndarray, out: np.ndarray | None = None, log_sum: bool | None = False
) -> np.ndarray | tuple[np.ndarray, float]:
    """Computes posterior density for vector of log-likelihood evaluated at each parameter component.

    I.e. if,
    log_x = [log(p_mat(obs_i | theta_0)), log(p_mat(obs_i | theta_1)),..., log(p_mat(obs_i | theta_{n-1}))],

    then returned value is,

    [p_mat(theta_0| obs_i),...,p_mat(theta_{n-1}|obs_i)],

    where,

    p_mat(theta_j| obs_i) = p_mat(obs_i| theta_j) / p_mat(obs_i).

    Args:
        log_x(ndarray): Numpy array of log-density values for each component/parameter value
            log_x = [log(p_mat(obs_i | theta_0)), log(p_mat(obs_i | theta_1)),..., log(p_mat(obs_i | theta_{n-1}))].
        out (Optional[ndarray]): Optional numpy array to store returned value.
        log_sum (Optional[bool]): If true returns Tuple with ([p_mat(obs_i|theta_j)], log(p_mat(obs_i))).
    Returns:
         Numpy array of posterior for each component/parameter value [p_mat(theta_0| obs_i),...,p_mat(theta_{n-1}|obs_i)].
         Optional tuple with ([p_mat(obs_i|theta_j)], log(p_mat(obs_i))) if log_sum true.
    """

    if out is None:
        rv = np.zeros(len(log_x))
    else:
        rv = out

    max_val = log_x.max()
    rv_sum = 0.0

    if isinf(max_val) or isnan(max_val):
        rv.fill(1.0 / float(len(log_x)))

    else:
        np.subtract(log_x, max_val, out=rv)
        np.exp(rv, out=rv)
        rv_sum = rv.sum()
        rv /= rv_sum
        rv_sum = np.log(rv_sum) + max_val

    if log_sum:
        return rv, rv_sum
    else:
        return rv


def log_posterior_sum(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Computes posterior density for vector of log-likelihood evaluated at each parameter component.

    I.e. if,

    log_x = [log(p_mat(obs_i | theta_0)), log(p_mat(obs_i | theta_1)),..., log(p_mat(obs_i | theta_{n-1}))],

    then returned value is a Tuple containing,

    [log(p_mat(theta_0| obs_i)),...,log(p_mat(theta_{n-1}|obs_i))] and log(p_mat(obs_i)),

    where, p_mat(theta_j| obs_i) = p_mat(obs_i| theta_j) / p_mat(obs_i).

    Args:
        x (np.ndarray): Numpy array of log-density values for each component/parameter value
            log_x = [log(p_mat(obs_i | theta_0)), log(p_mat(obs_i | theta_1)),..., log(p_mat(obs_i | theta_{n-1}))].
    Returns:
        Tuple of numpy array containing log-posterior for each component/parameter value
            [log(p_mat(theta_0| obs_i)),...,log(p_mat(theta_{n-1}|obs_i))], and log(p_mat(obs_i))). The log-posterior value is
            [-log(len(x)),...,-log(len(x))] if x contains a nan or -np.inf value.

    """

    max_val = x.max()
    if isinf(max_val) or isnan(max_val):
        return zeros(len(x)) - log(len(x)), -inf

    mass = log(exp(x - max_val).sum()) + max_val
    return x - mass, mass


def weighted_log_posterior(x: np.ndarray, w: np.ndarray) -> list[float]:
    """Computes weighted posterior density for vector of log-likelihood evaluated at each parameter component.

    I.e. if,
    x = [log(p_mat(obs_i | theta_0)), log(p_mat(obs_i | theta_1)),..., log(p_mat(obs_i | theta_{n-1}))], and
    w = [log(weight_0),log(weight_1),...,log(weight_{n-1})],

    then returned value is a list of floats,

    [log(p_mat(theta_0| obs_i))+log(weight_0),...,log(p_mat(theta_{n-1}|obs_i))+log(weight_{n-1})].

    Args:
        x (ndarray): Numpy array of log-density values for each component/parameter value
        w (ndarray): Numpy array of log weights for each parameter value.

    Returns:
        List[float] containing log-posterior for each component/parameter value
        [log(p_mat(theta_0| obs_i)),...,log(p_mat(theta_{n-1}|obs_i))].

    """
    max_val = -inf

    rv = [0.0] * len(x)

    for i in range(len(x)):
        r = w[i] + x[i]
        if r > max_val:
            max_val = r
        rv[i] = r

    if isinf(max_val) or isnan(max_val):  # every component impossible -> uniform posterior (matches log_posterior)
        return [-log(len(x))] * len(x)

    e_sum = 0.0
    for i in range(len(x)):
        e_sum += exp(rv[i] - max_val)

    mass = log(e_sum) + max_val

    for i in range(len(x)):
        rv[i] -= mass

    return rv


def weighted_log_posterior_sum(x: np.ndarray, w: np.ndarray) -> tuple[list[float], float]:
    """Computes weighted posterior density for vector of log-likelihood evaluated at each parameter component.

    I.e. if,

    x = [log(p_mat(obs_i | theta_0)), log(p_mat(obs_i | theta_1)),..., log(p_mat(obs_i | theta_{n-1}))], and
    w = [log(weight_0),log(weight_1),...,log(weight_{n-1})],

    then returned value is a Tuple of List[float] and float, containing

    [log(p_mat(theta_0| obs_i))+log(weight_0),...,log(p_mat(theta_{n-1}|obs_i))+log(weight_{n-1})], and
    log(p_mat(obs_i)),

    where, p_mat(theta_j| obs_i) = p_mat(obs_i| theta_j) / p_mat(obs_i).

    Args:
        x: (np.ndarray): numpy array of log-density values for each component/parameter value
            log_x = [log(p_mat(obs_i | theta_0)), log(p_mat(obs_i | theta_1)),..., log(p_mat(obs_i | theta_{n-1}))].
        w (np.ndarray): List[float] or numpy array of log weights for each parameter value.
    Returns:
        Tuple of List[float] containing log-posterior for each component/parameter value
        [log(p_mat(theta_0| obs_i)),...,log(p_mat(theta_{n-1}|obs_i))] and log(p_mat(obs_i).

    """
    max_val = -inf

    rv = [0.0] * len(x)

    for i in range(len(x)):
        r = w[i] + x[i]
        if r > max_val:
            max_val = r
        rv[i] = r

    if isinf(max_val) or isnan(max_val):  # every component impossible -> uniform posterior, zero evidence
        return [-log(len(x))] * len(x), -inf

    e_sum = 0.0
    for i in range(len(x)):
        e_sum += exp(rv[i] - max_val)

    mass = log(e_sum) + max_val

    for i in range(len(x)):
        rv[i] -= mass

    return rv, mass


# tuple[float[:, :, :], float[:], float]
def matrix_log_posteriors(x: np.ndarray, u_mat: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """

    :param x:
    :param u_mat:
    :param u:
    :return:
    """
    h = u_mat.shape[0]
    w = u_mat.shape[1]
    z = x.shape[1]

    row_posteriors = zeros((h, w, z))
    outer_posterior = zeros(h)
    outer_max = -inf

    for i in range(h):
        row_sum = zero

        for j in range(z):
            temp = u_mat[i, :] + x[:, j]
            inner_max = temp.max()
            if isinf(inner_max) or isnan(inner_max):  # this slice impossible -> uniform, no evidence
                row_posteriors[i, :, j] = 1.0 / w
                row_sum += -inf
                continue
            temp = exp(temp - inner_max)
            inner_sum = temp.sum()

            row_posteriors[i, :, j] = temp / inner_sum
            row_sum += log(inner_sum) + inner_max

        row_sum = row_sum + u[i]
        if row_sum > outer_max:
            outer_max = row_sum
        outer_posterior[i] = row_sum

    if isinf(outer_max) or isnan(outer_max):  # every row impossible -> uniform outer posterior, zero evidence
        return row_posteriors, zeros(h) + 1.0 / h, -inf

    outer_posterior = exp(outer_posterior - outer_max)
    outer_sum = outer_posterior.sum()
    outer_posterior /= outer_sum

    ll = log(outer_sum) + outer_max

    return row_posteriors, outer_posterior, ll


def row_choice(p_mat: np.ndarray, rng: np.random.RandomState | None) -> np.ndarray:
    """Vectorized choice call for varying sampling weights on contained in the rows of p_mat.

    N, S = p_mat.shape

     Choice is called on range [0,S), where the rows of p_mat are the sample weights.

     An N dim np.ndarray of ints is returned.

    Args:
        p_mat (np.ndarray): N by S matrix with weights
        rng (Optional[RandomState]): Set see for sampling.

    Returns:
        N dim numpy array of ints.

    """
    N, m = p_mat.shape
    u = rng.rand(N)

    bins = np.cumsum(p_mat, axis=1)
    rv = (u[:, None] >= bins[:, :-1]).sum(axis=1)

    return rv
