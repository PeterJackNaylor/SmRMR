import jax.numpy as np
from jax.random import PRNGKey, uniform
import pytest

from smrmr.association_measures import distance_corr, projection_corr, HSIC
from smrmr.association_measures.kernel_tools import get_kernel_function

key = PRNGKey(42)
X = uniform(key, shape=(10, 50))
Y = uniform(key, shape=(10,))

x_1 = np.array([35, 23, 47, 17, 10, 43, 9, 6, 28])
y_1 = np.array([30, 33, 45, 23, 8, 49, 12, 4, 31])

x_2 = np.array([4, 2.54, 3, 9])
y_2 = np.array([0, -12.54, 103, 91])


def almost_equal(a, b, name):
    return abs(a - b) == pytest.approx(0), f"value of {name} is {a}"


def test_projection_correlation():
    # sanity check
    for p in range(X.shape[1]):
        res = projection_corr.method(X[:, p], Y)
        assert np.isscalar(res)

    pc_ = projection_corr.method(x_1, y_1)
    ans = 0.65910995
    assert almost_equal(pc_, ans, "pc_")

    pc_ = projection_corr.method(x_2, y_2)
    ans = 0.37210423
    assert abs(pc_ - ans) == pytest.approx(0), f"value of pc_ is {pc_}"
    assert almost_equal(pc_, ans, "pc_")


def test_distance_correlation():
    # sanity check
    for p in range(X.shape[1]):
        res = distance_corr.method(X[:, p], Y)
        assert np.isscalar(res)

    # checked with https://rdrr.io/cran/Rfast/
    dc_ = distance_corr.method(x_1, y_1)
    ans = 0.9513721
    assert almost_equal(dc_, ans, "dc_")


def ref_hsic(X, Y, kernel, sigma):
    """
    From: https://github.com/wittawatj/fsic-test/blob/master/fsic/indtest.py
    Compute the biased estimator of HSIC as in Gretton et al., 2005.
    :param k: a Kernel on X
    :param l: a Kernel on Y
    """

    if X.shape[0] != Y.shape[0]:
        msg = "X and Y must have the same number of rows (sample size)"
        raise ValueError(msg)

    n = X.shape[0]

    kernel, kernel_params = get_kernel_function(kernel, nfeats=sigma)

    K = kernel(X, **kernel_params)
    L = kernel(Y, **kernel_params)
    Kmean = np.mean(K, 0)
    Lmean = np.mean(L, 0)
    HK = K - Kmean
    HL = L - Lmean
    # t = trace(KHLH)
    HKf = HK.flatten() / (n)
    HLf = HL.T.flatten() / (n)
    hsic = HKf.dot(HLf)

    return hsic


def test_hsic():
    ours = []
    ref = []
    for p in range(50):
        ours.append(
            float(
                HSIC.method(X[:, p], Y, kernel="gaussian", sigma=50) / X.shape[0] ** 2
            )
        )
        ref.append(ref_hsic(X[:, p], Y, kernel="gaussian", sigma=50))
    ours = np.array(ours)
    ref = np.array(ref)

    assert len(ours) == X.shape[1]
    assert all(ours >= 0)

    assert almost_equal(ref.sum(), ours.sum(), "our HSIC")

    # I got 0.09954543 with a R method..
    hsic_ = HSIC.method(x_1, y_1, kernel="gaussian", sigma=1) / x_1.shape[0] ** 2
    ans = 0.09528812
    assert almost_equal(hsic_, ans, "our HSIC")
