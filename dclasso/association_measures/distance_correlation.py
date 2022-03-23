# from dcor import distance_correlation as dc
import jax
from .am import AM
import jax.numpy as np

# import dcor
from jax import vmap


def distance_corr2(X, Y):
    """
    Computes the distance correlation between X and Y.
    Taken from pypi package dcor based on the paper:
    *Measuring and testing dependence by correlation of distances*
    by Gábor et Al (2007)
    Parameters
    ----------
    X : numpy array like object where the rows correspond to the samples
        and the columns to features.

    Y : numpy array like, of same size as X and one single output.

    Returns
    -------
    numpy array of size the number of input features of X
    which holds the distance correlation between each feature
    and Y.
    """
    n, d = X.shape
    if Y is not None:
        ny, nd = Y.shape

        assert n == ny
        assert nd == 1
    else:
        Y = X
        nd = d

    dc_stats = np.zeros((d, nd))

    for j in range(nd):
        for i in range(d):
            dc_stats[i, j] = dcor.distance_correlation(X[:, i], Y[:, j])

    return dc_stats


# @jax.jit
def p_distance(x: np.array, y: np.array, p: int) -> float:
    return np.sum((x - y) ** p)


def p_euclidian_distance(x: np.array, y: np.array, p: int) -> float:
    return p_distance(x, y, p) ** (1 / p)


# @jax.jit
def sqeuclidean_distance(x: np.array, y: np.array) -> float:
    return np.sum((x - y) ** 2)


# @jax.jit
def euclidean_distance(x: np.array, y: np.array) -> float:
    return np.sqrt(sqeuclidean_distance(x, y))


# @functools.partial(jax.jit, static_argnums=(0))
def distmat(x: np.ndarray, y: np.ndarray, p: int) -> np.ndarray:
    """distance matrix"""
    return vmap(lambda x1: vmap(lambda y1: p_euclidian_distance(x1, y1, p))(y))(x)
    # return vmap(lambda x1: vmap(lambda y1: euclidean_distance(x1, y1))(y))(x)


# pdist squareform
# @jax.jit
def pdist_p(x: np.ndarray, y: np.ndarray, p: int) -> np.ndarray:
    """squared euclidean distance matrix
    Notes
    -----
    This is equivalent to the scipy commands
    >>> from scipy.spatial.distance import pdist, squareform
    >>> dists = squareform(pdist(X, metric='sqeuclidean')
    """
    return distmat(x, y, p)


def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = np.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


def pdist_A(X, Y, p, unbiased=False):
    # https://dcor.readthedocs.io/en/latest/theory.html#properties
    n = X.shape[0]
    A = pdist_p(X, Y, p)
    if unbiased:
        A_1 = np.tile(A.sum(axis=1), (n, 1)) / (n - 2)
        A_0 = np.tile(A.sum(axis=0), (1, n)).reshape(n, n, order="F") / (n - 2)
        A = A - A_0 - A_1 + A.sum() / ((n - 1) * (n - 2))
        A = fill_diagonal(A, 0)
    else:
        A_1 = np.tile(A.mean(axis=1), (n, 1))
        A_0 = np.tile(A.mean(axis=0), (1, n)).reshape(n, n, order="F")
        mean = A.mean()
        A = A - A_0 - A_1 + mean
    return A


class dcor(AM):
    def method(self, X, Y, precompute=None, order_x=2, order_y=2, **args):
        unbiased = False
        # we could save some computation by saving Kx and Ky, because we could compute
        # them once instead of d*d.
        if precompute is None:
            Dx = pdist_A(X, X, p=order_x, unbiased=unbiased)
            Dy = pdist_A(Y, Y, p=order_y, unbiased=unbiased)
        else:
            Dx = precompute[0]
            Dy = precompute[1]

        dcor_ = np.multiply(Dx, Dy).sum()
        den = np.multiply(Dx, Dx).sum() * np.multiply(Dy, Dy).sum()

        dcor_ = dcor_ / (den**0.5)
        dcor_ = dcor_**0.5
        return dcor_


distance_corr = dcor()


def test_positivity():
    key = jax.random.PRNGKey(42)
    ns = 1000
    p = 100
    X = jax.random.normal(key, shape=(ns, p))
    Y = jax.random.normal(key, shape=(ns, p))
    for i in range(ns):
        assert distance_corr.method(X[i], Y[i]) >= 0
        assert 1 >= distance_corr.method(X[i], Y[i])


def random_value_test():
    X = np.array([4, 2.54, 3, 9])
    Y = np.array([0, -12.54, 103, 91])
    val = distance_corr.method(X, Y)
    print(val)
    assert 0.5822709405205625 == val
